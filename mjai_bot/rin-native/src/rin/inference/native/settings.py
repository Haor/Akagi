"""Strict settings for the FP32 native deployment contract."""
from dataclasses import dataclass,asdict,fields

@dataclass(frozen=True)
class Settings:
    model: str = "ppo400"
    device: str = "cuda:0"
    precision: str = "fp32"
    optimization_profile: str = "balanced"
    attention: str = "sdpa"
    cpu_threads: int = 4
    warmup: int = 3
    diagnostics: bool = False
    compile: bool = False
    tf32: bool = False

    @classmethod
    def parse(cls,value):
        unknown=set(value)-{f.name for f in fields(cls)}
        if unknown: raise ValueError(f"Unknown settings: {sorted(unknown)}")
        s=cls(**value)
        from .checkpoint import validate_model_id
        validate_model_id(s.model)
        if s.precision != "fp32": raise ValueError("Only FP32 is qualified for this deployment")
        if s.optimization_profile not in ("baseline","balanced"): raise ValueError("Unsupported optimization profile; fast is not qualified")
        if s.attention not in ("reference","sdpa"): raise ValueError("Unsupported attention")
        for k,low,high in (("cpu_threads",1,32),("warmup",1,100)):
            if type(getattr(s,k)) is not int or not low<=getattr(s,k)<=high: raise ValueError(f"{k} must be an integer in [{low},{high}]")
        for k in ("diagnostics","compile","tf32"):
            if type(getattr(s,k)) is not bool: raise ValueError(f"{k} must be boolean")
        if s.compile or s.tf32: raise ValueError("compile and TF32 are not qualified on this release")
        if s.device!="cpu" and s.device!="mps" and not (s.device.startswith("cuda:") and s.device[5:].isdigit()): raise ValueError("device must be cpu, mps or cuda:N")
        return s

    def effective(self):
        v=asdict(self)
        if self.optimization_profile=="balanced": v["attention"]="sdpa"
        return v

    def validate_qualification(self,root):
        del root
        if self.precision != "fp32" or self.compile or self.tf32:
            raise ValueError("This deployment requires FP32, disabled TF32 and no compilation")
