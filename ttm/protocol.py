# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, conint, validator


class MusicGeneration(bt.Synapse, BaseModel):
    """
    A class that transforms textual descriptions into music using machine learning models such as 
    'facebook/musicgen-medium' and 'facebook/musicgen-large'. Extends bt.Synapse for seamless integration into a 
    broader neural-based generative system.
    """
    text_input: str = Field(
        ...,
        title="Text Input",
        description="Textual directives or descriptions intended to guide the music generation process."
    )
    model_name: Optional[Literal['facebook/musicgen-medium', 'facebook/musicgen-large']] = Field(
        'facebook/musicgen-medium',
        title="Model Name",
        description="The machine learning model employed for music generation. Supported models: "
                    "'facebook/musicgen-medium', 'facebook/musicgen-large'."
    )
    music_output: Optional[List[bytes]] = Field(
        default=None,
        title="Music Output",
        description="The resultant music data, encoded as a list of bytes, generated from the text input."
    )
    duration: conint(gt=0) = Field(
        ...,
        title="Duration",
        description="The length of the generated music piece, specified in seconds. Must be greater than zero."
    )

    class Config:
        """ Configuration for validation on attribute assignment and strict data handling. """
        validate_assignment = True
        protected_namespaces = ()

    @validator('text_input')
    def text_input_not_empty(cls, value):
        """ Ensure the text input is not empty. """
        if not value or not value.strip():
            raise ValueError('Text input cannot be empty.')
        return value

    def deserialize(self) -> List[bytes]:
        """
        Processes and returns the music_output into a format ready for audio rendering or further analysis.
        """
        if not self.music_output:
            raise ValueError("No music output to deserialize.")
        return self.music_output
    