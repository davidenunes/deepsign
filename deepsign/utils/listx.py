def match_replace(elem, replacement_map):
    """ Expects an element and a set of rules that we can match
    the element against

    :param elem: the element to be match against
    :param replacement_map: an iterable set of rules in the form (bool_fn,elem_r) where
        - fn: elem -> bool
        - elem_r is the replacement of the given elem

    :return: elem if nothing matches
             the first replacement found if something matches
    """
    for replace_rule in replacement_map:
        (bool_fn,result) = replace_rule
        if bool_fn(elem):
            return result
    return elem
