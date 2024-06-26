from torchvision.models import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
)

from base_model import BaseModel


class ConvNext(BaseModel):
    def __init__(self, type, **kwargs):
        super(ConvNext, self).__init__(**kwargs)

        if type == "tiny":
            print('--------------tiny-------------------------')
            self.model = convnext_tiny(
                num_classes=kwargs["num_classes"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )
        elif type == "small":
            print('--------------small-------------------------')
            self.model = convnext_small(
                num_classes=kwargs["num_classes"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )
        elif type == "base":
            print('--------------base-------------------------')

            self.model = convnext_base(
                num_classes=1, #kwargs["num_classes"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )
        elif type == "large":
            print('--------------large-------------------------')

            self.model = convnext_large(
                num_classes=kwargs["num_classes"],
                stochastic_depth_prob=kwargs["stochastic_depth"],
            )

    def forward(self, x):
        return self.model(x)
