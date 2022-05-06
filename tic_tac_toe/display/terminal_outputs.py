def display():
    """
    Prints a display showing user where numerical inputs will be placed. Returns a list of '-' representing potential
    placement locations.

    :returns: List of ten elements. Nine are potential placements for 'X' or 'O'
    :rtype: list
    """
    display_fig = ['-', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    display_changes(display_fig)
    display_update = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    return display_update


def display_changes(table:list):
    """
    Prints a display showing the current values in index one through nine of the input table.

    :param tabel: A list of ten elements.
    :type table: list
    :returns: None
    """
    print(table[7] + "|" + table[8] + "|" + table[9] +
          "\n" + table[4] + "|" + table[5] + "|" + table[6] +
          "\n" + table[1] + "|" + table[2] + "|" + table[3])