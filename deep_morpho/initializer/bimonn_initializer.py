from copy import deepcopy
from enum import Enum
from typing import Dict, List, Tuple, Union

from .bisel_initializer import BiselInitializer
from .bise_initializer import (
    InitNormalIdentity, InitIdentity, InitKaimingUniform, InitIdentitySybise, InitNormalIdentitySybise,
    InitKaimingUniformSybise, InitBiseHeuristicWeights, InitBiseConstantVarianceWeights, InitSybiseHeuristicWeights,
    InitSybiseConstantVarianceWeights, InitSybiseConstantVarianceWeightsRandomBias, InitBiseEnum,
    InitSybiseHeuristicWeightsRandomBias, InitBiseEllipseWeights, InitBiseEllipseWeightsRoot,
    InitDualBiseConstantVarianceWeights, InitDualBiseConstantVarianceWeightsRandomBias,
    InitBiseConstantVarianceWeightsRandomBias, InitBiseConstantVarianceConstantWeights, InitDualBiseConstantVarianceConstantWeights,
    InitBiseConstantVarianceConstantWeightsRandomBias, InitDualBiseConstantVarianceConstantWeightsRandomBias,
    InitSybiseConstantVarianceConstantWeights, InitSybiseConstantVarianceConstantWeightsRandomBias
)


class InitBimonnEnum(Enum):
    IDENTICAL = 1
    INPUT_MEAN = 2



class BimonnInitializer:
    """The Bimonn initializer works in two steps. First, it generates intializers for each bisel. Then, it proposes
    another step of post initialization, to rework the initialization (ex: in case we need to adapt init to other layers).
    """
    def __init__(
        self,
        bise_init_method: Union[InitBiseEnum, List[InitBiseEnum]],
        bise_init_args: Union[Dict, List[Dict]],
        lui_init_method: Union[InitBiseEnum, List[InitBiseEnum]] = None,
        lui_init_args: Union[Dict, List[Dict]] = None,
    ):
        self.bise_init_method = bise_init_method
        self.bise_init_args = bise_init_args
        self.lui_init_method = lui_init_method if lui_init_method is not None else bise_init_method
        self.lui_init_args = lui_init_args if lui_init_args is not None else bise_init_args

    def generate_bisel_initializers(self, module) -> List[BiselInitializer]:
        res = []
        for layer_idx in range(len(module)):
            # attr_dict = {}

            # for attr in ["bise_init_method", "bise_init_args", "lui_init_method", "lui_init_args"]:
            #     attr_value = getattr(self, attr)
            #     if isinstance(attr_value, list):
            #         attr_value = attr_value[layer_idx]
            #     attr_dict[attr] = deepcopy(attr_value)

            # res.append(BiselInitializer(
            #     bise_initializer=self.get_init_class(attr_dict["bise_init_method"], module.atomic_element[layer_idx])(**attr_dict["bise_init_args"]),
            #     lui_initializer=self.get_init_class(attr_dict["lui_init_method"], module.atomic_element[layer_idx])(**attr_dict["lui_init_args"]),
            # ))
            res.append(self.generate_bisel_initializers_layer(module, layer_idx))

        return res

    def generate_bisel_initializers_layer(self, module, layer_idx: int):
        attr_dict = {}

        for attr in ["bise_init_method", "bise_init_args", "lui_init_method", "lui_init_args"]:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, list):
                attr_value = attr_value[layer_idx]
            attr_dict[attr] = deepcopy(attr_value)

        return BiselInitializer(
            bise_initializer=self.get_init_class(attr_dict["bise_init_method"], module.atomic_element[layer_idx])(**attr_dict["bise_init_args"]),
            lui_initializer=self.get_init_class(attr_dict["lui_init_method"], module.atomic_element[layer_idx])(**attr_dict["lui_init_args"]),
        )

    def post_initialize(self, module):
        pass

    @staticmethod
    def get_init_class(bise_init_method, atomic_element: str):
        if atomic_element == "bisel":
            if bise_init_method.value == InitBiseEnum.IDENTITY.value:
                return InitIdentity

            if bise_init_method.value == InitBiseEnum.NORMAL.value:
                return InitNormalIdentity

            elif bise_init_method.value == InitBiseEnum.KAIMING_UNIFORM.value:
                return InitKaimingUniform

            elif bise_init_method.value == InitBiseEnum.CUSTOM_HEURISTIC.value:
                return InitBiseHeuristicWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT.value:
                return InitBiseConstantVarianceWeights

            elif bise_init_method.value == InitBiseEnum.ELLIPSE.value:
                return InitBiseEllipseWeights

            elif bise_init_method.value == InitBiseEnum.ELLIPSE_ROOT.value:
                return InitBiseEllipseWeightsRoot

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_DUAL.value:
                return InitDualBiseConstantVarianceWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_DUAL_RANDOM_BIAS.value:
                return InitDualBiseConstantVarianceWeightsRandomBias

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS.value:
                return InitBiseConstantVarianceWeightsRandomBias

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS.value:
                return InitBiseConstantVarianceConstantWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL.value:
                return InitDualBiseConstantVarianceConstantWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS.value:
                return InitBiseConstantVarianceConstantWeightsRandomBias

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_DUAL_RANDOM_BIAS.value:
                return InitDualBiseConstantVarianceConstantWeightsRandomBias


        elif atomic_element == "sybisel":
            if bise_init_method.value == InitBiseEnum.IDENTITY.value:
                return InitIdentitySybise

            if bise_init_method.value == InitBiseEnum.NORMAL.value:
                return InitNormalIdentitySybise

            elif bise_init_method.value == InitBiseEnum.KAIMING_UNIFORM.value:
                return InitKaimingUniformSybise

            elif bise_init_method.value == InitBiseEnum.CUSTOM_HEURISTIC.value:
                return InitSybiseHeuristicWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT.value:
                return InitSybiseConstantVarianceWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_RANDOM_BIAS.value:
                return InitSybiseConstantVarianceWeightsRandomBias

            elif bise_init_method.value == InitBiseEnum.CUSTOM_HEURISTIC_RANDOM_BIAS.value:
                return InitSybiseHeuristicWeightsRandomBias

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS.value:
                return InitSybiseConstantVarianceConstantWeights

            elif bise_init_method.value == InitBiseEnum.CUSTOM_CONSTANT_CONSTANT_WEIGHTS_RANDOM_BIAS.value:
                return InitSybiseConstantVarianceConstantWeightsRandomBias


class BimonnInitInputMean(BimonnInitializer):
    def __init__(
        self,
        input_mean: float,
        bise_init_method: Union[InitBiseEnum, List[InitBiseEnum]],
        bise_init_args: Union[Dict, List[Dict]],
        lui_init_method: Union[InitBiseEnum, List[InitBiseEnum]] = None,
        lui_init_args: Union[Dict, List[Dict]] = None,
    ):
        self.input_mean = input_mean
        self.bise_init_method = bise_init_method
        self.bise_init_args = bise_init_args
        self.lui_init_method = lui_init_method if lui_init_method is not None else bise_init_method
        self.lui_init_args = lui_init_args if lui_init_args is not None else bise_init_args


    def generate_bisel_initializer_layer(self, module, layer_idx) -> List[BiselInitializer]:
        attr_dict = {}
        atomic_element = module.atomic_element[layer_idx]

        for attr in ["bise_init_method", "bise_init_args", "lui_init_method", "lui_init_args"]:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, list):
                attr_value = attr_value[layer_idx]
            attr_dict[attr] = deepcopy(attr_value)

        if layer_idx == 0:
            attr_dict["bise_init_args"]["input_mean"] = self.input_mean
        elif atomic_element == "bisel":
            attr_dict["bise_init_args"]["input_mean"] = attr_dict["bise_init_args"].get("input_mean", 0.5)
            attr_dict["lui_init_args"]["input_mean"] = attr_dict["lui_init_args"].get("input_mean", 0.5)
        elif atomic_element == "sybisel":
            attr_dict["bise_init_args"]["input_mean"] = attr_dict["bise_init_args"].get("input_mean", 0)
            attr_dict["lui_init_args"]["input_mean"] = attr_dict["lui_init_args"].get("input_mean", 0)

        return BiselInitializer(
            bise_initializer=self.get_init_class(attr_dict["bise_init_method"], atomic_element)(**attr_dict["bise_init_args"]),
            lui_initializer=self.get_init_class(attr_dict["lui_init_method"], atomic_element)(**attr_dict["lui_init_args"]),
        )
