import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    def forward_mapping(name: str, x: float):
        if name == 'square':
            f = x ** 2
        elif name == 'sin':
            f = np.sin(x)
        elif name == 'exp':
            f = np.exp(x)
        elif name == 'log':
            if x <= 0:
                raise ValueError("log requires x > 0")
            f = np.log(x)
        else:
            raise ValueError(f"unknown function: {name}")
        return f

    def gradient_mapping(name: str, z: float):
        if name == 'square':
            g = 2 * z
        elif name == 'sin':
            g = np.cos(z)          # 修正：不是 -cos
        elif name == 'exp':
            g = np.exp(z)
        elif name == 'log':
            if z <= 0:
                raise ValueError("log requires z > 0")
            g = 1 / z
        else:
            raise ValueError(f"unknown function: {name}")
        return g

    functions_reverse = functions[::-1]

    # 正向求值，得到每层的输入
    values = [x]
    for name in functions_reverse:
        values.append(forward_mapping(name, values[-1]))
    values.pop()          # 去掉最外层输出
    values = values[::-1] # 对齐到 functions 的顺序

    gradient = 1.0
    for name, z in zip(functions, values):
        gradient *= gradient_mapping(name, z)
    return gradient