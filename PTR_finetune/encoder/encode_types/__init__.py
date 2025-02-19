
from .encode_qa import encode_qa, data_func_qa

encode_config = {
    "qa": {
        "encode_func": encode_qa,
        "data_func": data_func_qa,
    }
}