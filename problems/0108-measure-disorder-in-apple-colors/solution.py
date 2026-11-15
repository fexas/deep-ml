def disorder(apples: list) -> float:
    """
    Compute the disorder in a basket of apples.
    """
    
    seened_lst = []
    count_lst = []

    for apple in apples:
        if seened_lst.count(apple) == 0:
            seened_lst.append(apple)
            count_lst.append(1)
        else:
            index = 0
            for seen in seened_lst:
                if apple == seen:
                    count_lst[index] += 1
                    break
                else:
                    index += 1
    
    count_sum  = 0
    for count in count_lst:
        count_sum += count
    
    if count_sum != 0:
        freq_lst = [count / count_sum for count in count_lst]
    else:
        freq_lst = [0 * count for count in count_lst]
    
    gini_lst = [freq * (1-freq) for freq in freq_lst]

    gini_index = 0
    for gini in gini_lst:
        gini_index += gini
    
    return gini_index
    

