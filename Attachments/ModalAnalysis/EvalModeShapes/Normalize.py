def Normalize_lst(lst):
    """
    Input:
    lst = 1-dimentional list
    
    Output:
    Normalizes list's between -1 (min val) and 1 (max val)
    """
    
    
    norm_lst = []
    max_val = max(lst)
    min_val = min(lst)
    for i in range(len(lst)):
        val = 2*(lst[i] - min_val) / (max_val - min_val) -1
        norm_lst.append(val)

    return norm_lst