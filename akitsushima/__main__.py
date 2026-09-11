"""Enables ``python -m akitsushima ...`` as an alternative to the
installed ``akitsushima`` console script.
"""
import sys

from .cli.parser import main

if __name__ == "__main__":
    sys.exit(main())
