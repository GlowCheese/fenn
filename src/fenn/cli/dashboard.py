"""fenn dashboard — launch the Fenn log-browser web UI."""


def execute(args) -> None:
    """Import and start the dashboard server directly from the installed package."""
    try:
        from fenn.dashboard.app import run
    except ImportError as exc:
        raise SystemExit(
            "ERROR: Could not import the Fenn dashboard.\n"
            "Make sure Flask is installed:  pip install flask\n"
            f"Details: {exc}"
        )

    try:
        run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            log_dirs=args.log_dir,
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
