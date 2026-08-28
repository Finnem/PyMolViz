def sanitize_pymol_string(string):
    return string.replace(" ", "_").replace(",", "_").replace("(", "_").replace(")","_").replace("[", "_").replace("]", "_")