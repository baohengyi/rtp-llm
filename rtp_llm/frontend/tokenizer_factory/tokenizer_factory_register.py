import inspect
from typing import Any, Dict, List, Type, Union

_tokenizer_factory: Dict[str, Type[Any]] = {}


def _is_same_source_class(existing: Any, incoming: Any) -> bool:
    """True iff ``existing`` and ``incoming`` are the SAME class definition
    loaded under different module identities.

    Phase-25 namespace merge extends ``rtp_llm.__path__`` with the sibling
    ``internal_source/rtp_llm`` tree, so a source file such as
    ``internal_source/rtp_llm/tokenizers/flot_tokenizer.py`` can be imported
    under both ``rtp_llm.tokenizers.flot_tokenizer`` (relative import from
    ``rtp_llm.tokenizers.internal_init``) and
    ``internal_source.rtp_llm.tokenizers.flot_tokenizer`` (full-prefix import
    from ``internal_source/rtp_llm/models/flot_vl.py``). Python treats these
    as two distinct module objects, so the top-level ``class FlotTokenizer``
    executes twice and produces two class objects that fail ``is`` / ``==``
    identity checks even though they share source. We recognize this case by
    matching ``__qualname__`` AND source file path.
    """
    if existing is incoming:
        return True
    if not (inspect.isclass(existing) and inspect.isclass(incoming)):
        return False
    if getattr(existing, "__qualname__", None) != getattr(
        incoming, "__qualname__", None
    ):
        return False
    try:
        return inspect.getsourcefile(existing) == inspect.getsourcefile(incoming)
    except (TypeError, OSError):
        return False


def register_tokenizer(name: Union[str, List[str]], tokenizer: Any):
    global _tokenizer_factory
    if isinstance(name, List):
        for n in name:
            register_tokenizer(n, tokenizer)
    else:
        if name in _tokenizer_factory and _tokenizer_factory[name] != tokenizer:
            if _is_same_source_class(_tokenizer_factory[name], tokenizer):
                # Phase-25 namespace merge: same source class re-imported under
                # a different module path. Keep the first-registered entry;
                # both class objects have identical behaviour.
                return
            raise Exception(
                f"try register model {name} with type {_tokenizer_factory[name]} and {tokenizer}, confict!"
            )
        _tokenizer_factory[name] = tokenizer
