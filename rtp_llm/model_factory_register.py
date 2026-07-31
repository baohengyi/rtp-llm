import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Type

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

_model_factory: Dict[str, Type[Any]] = {}


def _is_same_source_class(existing: Any, incoming: Any) -> bool:
    """True iff ``existing`` and ``incoming`` are the SAME class definition
    loaded under different module identities.

    Phase-25 namespace merge extends ``rtp_llm.__path__`` with the sibling
    ``internal_source/rtp_llm`` tree, so ``internal_source/rtp_llm/models/X.py``
    can be imported under both ``rtp_llm.models.X`` (via the extended path) and
    ``internal_source.rtp_llm.models.X`` (via full-prefix imports still present
    in a handful of internal_source files). Python creates two distinct
    module objects, so the top-level class definition executes twice — the
    resulting class objects fail identity/equality checks even though they
    share source. Recognize this case by matching ``__qualname__`` AND source
    file path so duplicate registrations from the same source file are
    tolerated while genuinely different classes still raise.
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


def register_model(
    name: str,
    model_type: Any,
    support_architectures: List[str] = [],
    support_hf_repos: List[str] = [],
):
    global _model_factory
    if name in _model_factory and _model_factory[name] != model_type:
        if not _is_same_source_class(_model_factory[name], model_type):
            raise Exception(
                f"try register model {name} with type {_model_factory[name]} and {model_type}, confict!"
            )
        # Phase-25 namespace merge: same class re-imported under a different
        # module path. Skip the duplicate registration (including architecture
        # / repo re-registration below, which would compare the string model
        # name against itself and be a no-op anyway).
        return
    _model_factory[name] = model_type

    for architecture in support_architectures:
        register_hf_architecture(architecture, name)

    for repo in support_hf_repos:
        register_hf_repo(repo, name)


_hf_architecture_2_ft = {}


def register_hf_architecture(name: str, model_type: str):
    global _hf_architecture_2_ft
    if name in _hf_architecture_2_ft and _hf_architecture_2_ft[name] != model_type:
        raise Exception(
            f"try register model {name} with type {_hf_architecture_2_ft[name]} and {model_type}, confict!"
        )
    logging.debug("registerhf_architecture: %s -> %s", name, model_type)
    _hf_architecture_2_ft[name] = model_type


_hf_repo_2_ft = {}


def register_hf_repo(name: str, model_type: str):
    global _hf_repo_2_ft
    if name in _hf_repo_2_ft and _hf_repo_2_ft[name] != model_type:
        raise Exception(
            f"try register model {name} with type {_hf_repo_2_ft[name]} and {model_type}, confict!"
        )
    logging.debug("register_hf_repo: %s -> %s", name, model_type)
    _hf_repo_2_ft[name] = model_type


class ModelDict:
    @staticmethod
    def get_ft_model_type_by_hf_repo(repo: str) -> Optional[str]:
        global _hf_repo_2_ft
        model_type = _hf_repo_2_ft.get(repo, None)
        logging.debug("get hf_repo model type: %s, %s", repo, model_type)
        return model_type

    @staticmethod
    def get_ft_model_type_by_hf_architectures(architecture):
        global _hf_architecture_2_ft
        model_type = _hf_architecture_2_ft.get(architecture, None)
        logging.debug("get architectur model type: %s, %s", architecture, model_type)
        return model_type

    @staticmethod
    def get_ft_model_type_by_config(config: Dict[str, Any]) -> Optional[str]:
        if config.get("architectures", []):
            # hack for ChatGLMModel: chatglm and chatglm2 use same architecture
            architecture = config.get("architectures")[0]
            if architecture in ["ChatGLMModel", "ChatGLMForConditionalGeneration"]:
                _name_or_path = config.get("_name_or_path", "")
                if (
                    not config.get("multi_query_attention", False)
                    or "chatglm-6b" in _name_or_path
                ):
                    return "chatglm"
                elif "chatglm3" in _name_or_path:
                    return "chatglm3"
                elif "glm-4-" in _name_or_path:
                    return "chatglm4"
                elif "glm-4v" in _name_or_path:
                    return "chatglm4v"
                else:
                    return "chatglm2"
            if architecture == "QWenLMHeadModel":
                if config.get("visual"):
                    if config["visual"].get("layers"):
                        return "qwen_vl"
                    else:
                        return "qwen_vl_1b8"
            if architecture == "BaichuanForCausalLM":
                vocab_size = config.get("vocab_size", 64000)
                if vocab_size == 125696:
                    return "baichuan2"
                else:
                    return "baichuan"
            if architecture == "GPTNeoXForCausalLM":
                vocab_size = config.get("vocab_size", 50432)
                if vocab_size == 250752:
                    return "gpt_neox_13b"
                else:
                    return "gpt_neox"
            return ModelDict.get_ft_model_type_by_hf_architectures(architecture)
        else:
            logging.warning(f"config have no architectures: {config}")
        return None
