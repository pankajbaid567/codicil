import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "codicil"
FORBIDDEN_FOR_DOMAIN = {"httpx", "fastapi", "onnxruntime", "pydantic_settings", "codicil.adapters", "codicil.application", "codicil.api", "codicil.settings"}
FORBIDDEN_FOR_APPLICATION = {
    "httpx",
    "fastapi",
    "onnxruntime",
    "codicil.api",
    "codicil.adapters.onnx_embedder",
    "codicil.adapters.onnx_reranker",
    "codicil.adapters.openai_compat",
}


def imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def violations(layer: str, forbidden: set[str]) -> list[str]:
    bad = []
    for f in (SRC / layer).glob("*.py"):
        for imp in imports(f):
            if any(imp == b or imp.startswith(b + ".") for b in forbidden):
                bad.append(f"{f.name} imports {imp}")
    return bad


def test_domain_is_pure():
    assert violations("domain", FORBIDDEN_FOR_DOMAIN) == []


def test_application_depends_on_ports_not_adapters():
    # `index.py` imports the rbi_html parser lazily inside a function for the build use case; everything else must use ports only
    assert violations("application", FORBIDDEN_FOR_APPLICATION) == []
