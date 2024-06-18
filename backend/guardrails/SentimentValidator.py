from typing import Any, Callable, Dict, Optional, Union
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from SentimentPredictor import SentimentPredictor

@register_validator(name="my_validator", data_type="string")
class SentimentValidator(Validator):
    def __init__(
        self,
        # threshold: float = 0.5,
        # validation_method: str = "sentence",
        # device: Optional[Union[str, int]] = "cpu",
        # model_name: Optional[str] = "unbiased-small", 
        on_fail: Union[Callable[..., Any], None] = None,
        **kwargs,
    ):
        """Initializes a new instance of the MyValidator class.
        Args:
            arg_1 (str): FIXME: Describe the purpose of this argument.
            on_fail`** *(str, Callable)*: The policy to enact when a validator fails.  If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.
        """
        super().__init__(on_fail, **kwargs)
        # self._threshold = float(threshold)
        # Define the model, pipeline and labels
        self._model = SentimentPredictor()

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validates that {fill in how you validator interacts with the passed value}.
        
        Args:
            value (Any): The value to validate.
            metadata (Dict): The metadata to validate against.

            FIXME: Add any additional args you need here in metadata.
            | Key | Description |
            | --- | --- |
            | a | b |
        """
        # 아래 클래스 포함될 시 fail
        # {'labels': ['우울한', '질투하는', '상처', '열등감', '염세적인', '괴로워하는', '버려진', '낙담한', '슬픔', '충격 받은']
        # , 'classes': [14, 31, 30, 44, 16, 38, 39, 18, 10, 34]}
        restricted_classes = [14, 31, 30, 44, 16, 38, 39, 18, 10, 34]
        result = self._model.predict(value)

        for c in result.get('classes'):
            if c in restricted_classes:
                print('result', result)
                return FailResult(
                    error_message="including restricted classes",               
                    fix_value="우울해하지마!",
                )
        return PassResult()