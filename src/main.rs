use clap::Parser;

/// vipune - A minimal memory layer for AI agents
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    // Placeholder for CLI arguments
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        println!("vipune initialized");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::parse_from(&["vipune", "--verbose"]);
        assert!(cli.verbose);
    }

    #[test]
    fn test_cli_default() {
        let cli = Cli::parse_from(&["vipune"]);
        assert!(!cli.verbose);
    }
}
