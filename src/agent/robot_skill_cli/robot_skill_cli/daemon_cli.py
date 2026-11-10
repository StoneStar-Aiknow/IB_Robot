"""Legacy deployment arguments for the neutral resident transport command."""

import argparse


def main(argv=None):
    from robot_client.transports.resident import main as serve

    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--config-path")
    group.add_argument("--config-name")
    args, remaining = parser.parse_known_args(argv)
    kwargs = None
    if args.config_path or args.config_name:
        from robot_client.connection import bridge_kwargs
        from robot_config.config_path import resolve_robot_config_path
        from robot_config.loader import load_robot_config_dict
        from robot_skill_cli.catalog import _gateway_transport

        path = resolve_robot_config_path(config_name=args.config_name, config_path=args.config_path)
        config = load_robot_config_dict(path)
        kwargs = bridge_kwargs(_gateway_transport(config.get("embodied", {}), robot_id=config.get("name", "")))
    return serve(remaining, prewarm_kwargs=kwargs)
