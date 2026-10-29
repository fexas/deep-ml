def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])
    product = []

    if cols_a != rows_b:
        return -1

    for i in range(rows_a):
        new_row = []
        for j in range(cols_b):
            element = 0
            for k in range(cols_a):
                element += a[i][k] * b[k][j]
            new_row.append(element)
            if j == cols_b - 1:
                product.append(new_row)
    return product