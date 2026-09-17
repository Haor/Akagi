"""Explicit Flax-equivalent visible actor; shared by CUDA, CPU and MPS.

Weights retain their Flax names in safetensors. Dense and DenseGeneral axes are mapped
once at load time; embeddings, normalization scales and tokens are not transposed.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def semantic_tile_features(**s):
    """Pure NumPy public-feature adapter, outside the neural network boundary."""
    def histogram(tiles, red=False):
        tiles=np.asarray(tiles,dtype=np.int32)
        valid=(tiles>=0)&(tiles<37)
        if red: valid &= tiles>=34
        base=tiles.copy()
        for r,b in ((34,4),(35,13),(36,22)): base=np.where(base==r,b,base)
        return (np.eye(34,dtype=np.float32)[np.clip(base,0,33)]*valid[...,None]).sum(axis=-2)
    hand=np.asarray(s["hand_counts_red37"],dtype=np.float32)
    base=hand[:,:34].copy()
    red=np.zeros_like(base)
    for r,b in ((34,4),(35,13),(36,22)):
        base[:,b]+=hand[:,r]
        red[:,b]=hand[:,r]
    draw=np.asarray(s["last_draw"])[:,None]
    disc=np.asarray(s["discards"])
    dora=histogram(s["dora_indicators"])
    meld=histogram(np.asarray(s["meld_tiles"]).reshape(len(hand),4,-1))
    uncalled=histogram(np.where(s["discard_called"],-1,disc))
    groups=[(base/4)[:,None,:],red[:,None,:],histogram(draw)[:,None,:],histogram(draw,True)[:,None,:],(dora/5)[:,None,:],histogram(s["dora_indicators"],True)[:,None,:],histogram(disc)/4,histogram(np.where(s["discard_tsumogiri"],disc,-1))/4,np.clip(histogram(np.where(s["discard_riichi"],disc,-1)),0,1),histogram(np.where(s["discard_called"],disc,-1))/4,meld/4,(np.clip(base+dora+uncalled.sum(1)+meld.sum(1),0,4)/4)[:,None,:]]
    return np.concatenate(groups,axis=1).swapaxes(1,2).astype(np.float32)

class SemanticActorTorch(nn.Module):
    def __init__(self, arrays, config, *, device="cpu", precision="fp32", attention="reference"):
        super().__init__()
        self.config=config["model"]
        self.compute_dtype={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[precision]
        self.attention=attention
        self.weights=nn.ParameterDict()
        for name,array in arrays.items():
            value=torch.from_numpy(array.copy())
            if name.endswith("/kernel"):
                if value.ndim==3:
                    if "/attention/out/" in name: value=value.reshape(-1,value.shape[-1])
                    else: value=value.reshape(value.shape[0],-1)
                value=value.T.contiguous()
            elif name.endswith("/bias") and value.ndim==2:
                value=value.flatten()
            dtype=torch.float32 if name.endswith("/scale") or name.startswith("action_scorer/output/") or name.endswith("kind_bias") else self.compute_dtype
            self.weights[name.replace("/",".").replace(".","__")]=nn.Parameter(value.to(device=device,dtype=dtype),requires_grad=False)
        self.register_buffer("tile_ids",torch.arange(34,device=device))
        self.register_buffer("flag_positions",torch.arange(16,device=device))
        self.eval()

    def w(self,p): return self.weights[p.replace("/","__")]

    def dense(self,x,p,fp32=False):
        dtype=torch.float32 if fp32 else self.compute_dtype
        return F.linear(x.to(dtype),self.w(p+"/kernel"),self.w(p+"/bias"))

    def norm(self,x,p):
        # Flax RMSNorm accumulates variance and applies scale in float32.
        z=x.float()
        return (z*torch.rsqrt((z*z).mean(-1,keepdim=True)+1e-6)*self.w(p+"/scale")).to(self.compute_dtype)

    def embed(self,values,p,safe=True):
        w=self.w(p+"/embedding")
        ids=values.long()
        if safe: ids=(ids+1).clamp(0,len(w)-1)
        return F.embedding(ids,w)

    def swiglu(self,x,p):
        gate,value=self.dense(x,p+"/in").chunk(2,-1)
        return self.dense(F.silu(gate)*value,p+"/out")

    def residual(self,x,p):
        return x+self.swiglu(self.norm(x,p+"/norm"),p+"/swiglu")

    def transformer(self,x,p,heads,mask=None):
        z=self.norm(x,p+"/attention_norm")
        b,t,w=z.shape
        q,k,v=[self.dense(z,p+"/attention/"+name).reshape(b,t,heads,w//heads).transpose(1,2) for name in ("query","key","value")]
        if self.attention=="sdpa":
            # Flax scales Q before the contraction. Preserve its rounding point.
            q=q/math.sqrt(w//heads)
            bias=None if mask is None else torch.where(mask,0.0,torch.finfo(q.dtype).min).to(q.dtype)
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.0,scale=1.0)
        else:
            scores=(q/math.sqrt(w//heads))@k.transpose(-1,-2)
            if mask is not None: scores=scores.masked_fill(~mask,torch.finfo(scores.dtype).min)
            out=torch.softmax(scores,dim=-1)@v
        out=out.transpose(1,2).reshape(b,t,w)
        x=x+self.dense(out,p+"/attention/out")
        return x+self.swiglu(self.norm(x,p+"/mlp_norm"),p+"/swiglu")

    def consumed(self,values,p):
        e=self.embed(values,p)
        mask=(values>=0).unsqueeze(-1)
        count=mask.sum(-2).clamp_min(1).float()
        # Integer sqrt promotes to float32 in the fixed Flax actor.
        return torch.where(mask,e,0).sum(-2).float()/torch.sqrt(count)

    def forward(self,tile_features,history,history_mask,global_features,candidates,candidate_mask,*,intermediates=False):
        c=self.config
        traces={}
        tokens=self.dense(tile_features,"semantic_projection")
        traces["semantic_projection"]=tokens
        ids=self.tile_ids
        suits=torch.where(ids<27,ids//9,3)
        ranks=torch.where(ids<27,ids%9,9)
        rel=torch.where(ids>=27,2,torch.where((ranks==0)|(ranks==8),1,0))
        for name,values in (("tile",ids),("suit",suits),("rank",ranks),("terminal_honor",rel)):
            tokens=tokens+self.embed(values,f"tile_encoder/{name}_embedding",False)[None]
        tokens=torch.cat((self.w("tile_encoder/state_token").expand(len(tokens),-1,-1),tokens),1)
        for i in range(c["tile_layers"]):
            tokens=self.transformer(tokens,f"tile_encoder/block_{i}",c["tile_heads"])
            if intermediates: traces[f"tile_block_{i}"]=tokens
        tile=self.norm(tokens[:,0],"tile_encoder/output_norm")
        traces["tile_encoder"]=tile
        h=None
        for name in ("event_type","actor","target","tile","tsumogiri","meld_orientation","round_position","recency"):
            e=self.embed(history[name],f"history_encoder/{name}_embedding")
            h=e if h is None else h+e
        h=h+self.consumed(history["consumed_tiles"],"history_encoder/consumed_tile_embedding")
        h=torch.where(history_mask[...,None],h,0)
        h=torch.cat((self.w("history_encoder/history_token").expand(len(h),-1,-1),h),1)
        hm=torch.cat((torch.ones((len(h),1),device=h.device,dtype=torch.bool),history_mask.bool()),1)
        attention_mask=hm[:,None,:,None]&hm[:,None,None,:]
        for i in range(c["history_layers"]):
            h=self.transformer(h,f"history_encoder/block_{i}",c["history_heads"],attention_mask)
            if intermediates: traces[f"history_block_{i}"]=h
        hist=self.norm(h,"history_encoder/output_norm")[:,0]
        traces["history_encoder"]=hist
        glob=self.dense(F.silu(self.dense(global_features,"global_encoder/hidden")),"global_encoder/output")
        traces["global_encoder"]=glob
        state=self.dense(torch.cat((tile,hist,glob),-1),"fusion_projection")
        for i in range(c["fusion_layers"]): state=self.residual(state,f"fusion_block_{i}")
        state=self.norm(state,"state_norm")
        traces["state_norm"]=state
        fields=[self.embed(candidates[key],f"action_encoder/{name}_embedding") for name,key in (("kind","kind"),("tile","tile"),("called_tile","called_tile"),("target","relative_target"),("from_draw","from_draw"),("red_mask","red_tile_mask"),("phase","decision_phase"))]
        fields.append(self.consumed(candidates["consumed_tiles"],"action_encoder/consumed_tile_embedding"))
        bits=(candidates["rule_flags"].long()[...,None]>>self.flag_positions)&1
        fields.append(self.w("action_encoder/rule_flag_embedding")[self.flag_positions,bits].sum(-2))
        actions=self.dense(torch.cat(fields,-1),"action_encoder/input_projection")
        for i in range(c["action_layers"]): actions=self.residual(actions,f"action_encoder/block_{i}")
        actions=self.norm(actions,"action_encoder/output_norm")
        traces["action_encoder"]=actions
        hidden=F.silu(self.dense(state,"action_scorer/state")[:,None]+self.dense(actions,"action_scorer/action")+self.dense(state[:,None]*actions,"action_scorer/interaction"))
        logits=self.dense(hidden,"action_scorer/output",True).squeeze(-1)+self.w("action_scorer/kind_bias")[candidates["kind"].long().clamp(0,10)]
        logits=logits.masked_fill(~candidate_mask.bool(),torch.finfo(torch.float32).min)
        return (logits,traces) if intermediates else logits

def tensor_inputs(inputs,device):
    if "semantic" in inputs:
        inputs={**inputs,"tile_features":semantic_tile_features(**inputs["semantic"])}
        inputs.pop("semantic")
    return {k:({n:torch.as_tensor(np.asarray(v),device=device) for n,v in value.items()} if isinstance(value,dict) else torch.as_tensor(np.asarray(value),device=device)) for k,value in inputs.items()}

class Predictor:
    def __init__(self, model_path, *, device="cuda:0", precision="fp32", attention="reference",cpu_threads=4):
        from .checkpoint import load_actor_arrays
        if str(device).startswith("cuda:"):
            index=int(str(device).split(":",1)[1])
            if not 0<=index<torch.cuda.device_count(): raise RuntimeError(f"Requested CUDA device unavailable: {device}")
        self.device=torch.device(device)
        if self.device.type=="cuda":
            if not torch.cuda.is_available() or (self.device.index or 0)>=torch.cuda.device_count(): raise RuntimeError(f"Requested CUDA device unavailable: {device}")
        elif self.device.type=="mps":
            import os
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK","0")!="0": raise RuntimeError("MPS CPU fallback must be disabled")
            if not torch.backends.mps.is_available(): raise RuntimeError("Requested MPS device unavailable")
        elif self.device.type!="cpu": raise ValueError("Device must be cpu, cuda:N, or mps")
        torch.set_num_threads(cpu_threads)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        arrays,self.config,self.manifest=load_actor_arrays(model_path)
        self.actor=SemanticActorTorch(arrays,self.config,device=self.device,precision=precision,attention=attention)
        self.effective={"device":str(self.device),"precision":precision,"attention":attention,"cpu_threads":cpu_threads,"tf32":False,"compile":False,"parameter_device":str(next(self.actor.parameters()).device),"gpu":torch.cuda.get_device_name(self.device) if self.device.type=="cuda" else str(self.device)}

    @torch.inference_mode()
    def predict(self,inputs):
        tensors=tensor_inputs(inputs,self.device)
        return self.actor(**tensors).float().cpu().numpy()
