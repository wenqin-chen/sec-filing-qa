"""``python -m secqa`` runs the same Typer application as the ``secqa`` console script."""

from secqa.cli import app

if __name__ == "__main__":
    app()
