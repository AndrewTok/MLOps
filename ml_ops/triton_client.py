from functools import lru_cache

import numpy as np
from tritonclient.http import (
    InferenceServerClient,
    InferInput,
    InferRequestedOutput,
)
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton_infer(input: np.ndarray):
    triton_client = get_client()

    triton_input = InferInput(
        name="IRIS_FEATURES",
        shape=input.shape,
        datatype=np_to_triton_dtype(input.dtype),
    )

    triton_input.set_data_from_numpy(input, binary_data=True)

    triton_output = InferRequestedOutput("CLASS_PROBS", binary_data=True)
    query_response = triton_client.infer(
        "simple_model", [triton_input], outputs=[triton_output]
    )

    output = query_response.as_numpy("CLASS_PROBS")

    return output


def main():
    inpupts = np.array(
        [
            [4.6000, 3.4000, 1.4000, 0.3000],
            [6.2000, 2.2000, 4.5000, 1.5000],
            [4.8000, 3.4000, 1.9000, 0.2000],
            [7.9000, 3.8000, 6.4000, 2.0000],
            [5.4000, 3.9000, 1.7000, 0.4000],
        ],
        dtype='float32',
    )

    outs = np.array(
        [
            [0.8345104, 0.09270251, 0.07278708],
            [0.17694306, 0.35782117, 0.4652358],
            [0.73796207, 0.14852425, 0.11351372],
            [0.06353826, 0.14495152, 0.7915102],
            [0.81613696, 0.10188907, 0.08197398],
        ],
        dtype='float32',
    )

    triton_out = call_triton_infer(inpupts)
    print(triton_out)
    print("Mean diff: " + str(np.mean(np.abs(outs - triton_out))))


if __name__ == "__main__":
    main()
