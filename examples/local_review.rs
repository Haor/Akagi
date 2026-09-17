//! Run the same offline review pipeline as the UI against a local MJAI file.
use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use tokio::process::Command;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    bot_dir: PathBuf,
    #[arg(long)]
    python: PathBuf,
    #[arg(long)]
    uv: Option<PathBuf>,
    #[arg(long)]
    log: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 0)]
    seat: u8,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let bot_dir = args.bot_dir.canonicalize()?;
    let python = args.python.canonicalize()?;
    let events = std::fs::read_to_string(&args.log)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<akagi::schema::MjaiEvent>, _>>()?;
    let manifest = akagi::bot::Manifest::load(&bot_dir)?.context("bot manifest is missing")?;
    let settings = akagi::bot::manifest::load_values(&bot_dir, &manifest)?;
    let settings_file = tempfile::NamedTempFile::new()?;
    serde_json::to_writer(settings_file.as_file(), &settings)?;
    let runtime = akagi::bot::PythonRuntime::from_paths(
        python.clone(),
        args.uv
            .clone()
            .unwrap_or_else(|| PathBuf::from("unused-uv")),
        akagi::bot::RuntimeMode::System,
    );
    let mut command = if args.uv.is_some() {
        runtime.ensure_synced(&bot_dir).await?;
        runtime.command_for(&bot_dir, &["bot.py"])
    } else {
        let mut command = Command::new(&python);
        command
            .current_dir(&bot_dir)
            .arg("bot.py")
            .env_remove("PYTHONHOME")
            .env_remove("PYTHONPATH");
        command
    };
    command
        .arg(args.seat.to_string())
        .env("AKAGI_BOT_CONFIG", settings_file.path());
    let mut runner = akagi::bot::SubprocessBot::spawn_with_command(
        command,
        runtime,
        &bot_dir,
        args.seat,
        akagi::event_bus::notify_bus(),
    )
    .await?;
    let result = akagi::history::local_review::replay(
        &mut runner,
        &events,
        args.seat,
        4,
        "local-file".into(),
        manifest.bot.name,
        |_, _| {},
    )
    .await?;
    serde_json::to_writer(std::fs::File::create(&args.output)?, &result)?;
    println!(
        "Reviewed {} events, {} decisions, {} compared, {} matched",
        result.event_count,
        result.decisions.len(),
        result.compared,
        result.matched
    );
    Ok(())
}
