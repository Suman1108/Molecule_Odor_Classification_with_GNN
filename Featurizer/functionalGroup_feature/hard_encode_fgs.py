from rdkit import Chem

# Define SMARTS patterns for all 40 functional groups
FG_SMARTS_PATTERNS = {
    'Acid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),               # Carboxylic acid -COOH
    'Acetamide': Chem.MolFromSmarts('NC(=O)C'),                   # Acetamide group
    'Alcohols': Chem.MolFromSmarts('[OX2H]'),                    # Alcohol -OH
    'Acetyl': Chem.MolFromSmarts('CC(=O)[#6]'),                   # Acetyl group
    'Aldehydes': Chem.MolFromSmarts('[CX3H1](=O)[#6]'),          # Aldehyde group
    'Alkanes': Chem.MolFromSmarts('[CX4]'),                       # Alkane carbons (sp3)
    'Amide': Chem.MolFromSmarts('C(=O)N'),                        # Amide group
    'Amine': Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]'),          # Primary/secondary amines not attached to carbonyl
    'Bicyclic': Chem.MolFromSmarts('[$([R2]),$([R3])]'),          # Atoms in 2 or 3 rings (approx.)
    'Cyclic': Chem.MolFromSmarts('[$([R])]'),                     # Any ring atom
    'Carbonyl': Chem.MolFromSmarts('[CX3]=O'),                    # Carbonyl group C=O
    'Esters': Chem.MolFromSmarts('C(=O)O[C]'),                    # Ester group
    'CarboxylicAcid': Chem.MolFromSmarts('[CX3](=O)[OX2H1]'),     # Same as Acid
    'Ethers': Chem.MolFromSmarts('[OD2]([#6])[#6]'),              # Ether -O-
    'Cyclopropyl': Chem.MolFromSmarts('C1CC1'),                   # Cyclopropyl ring
    'Furan': Chem.MolFromSmarts('c1ccoc1'),                       # Furan ring
    'Hydrocarbons': Chem.MolFromSmarts('[CX4H]'),                 # Alkane carbons with hydrogens
    'Ethoxy': Chem.MolFromSmarts('OCC'),                          # Ethoxy group
    'Imino': Chem.MolFromSmarts('[NX2]=C'),                       # Imino group
    'Ketones': Chem.MolFromSmarts('C(=O)[#6]'),                   # Ketone group
    'Lactone': Chem.MolFromSmarts('O=C1OC=CC1'),                  # Approximate lactone ring
    'N-Compounds': Chem.MolFromSmarts('[NX3]'),                   # Any nitrogen with 3 connection
    'Oximes': Chem.MolFromSmarts('C=NO'),                         # Oxime group
    'Methoxy': Chem.MolFromSmarts('OC'),                          # Methoxy group
    'Oxirane': Chem.MolFromSmarts('C1OC1'),                       # Epoxide ring
    'Phenol': Chem.MolFromSmarts('c1ccc(cc1)O'),                  # Phenol group
    'Pyran': Chem.MolFromSmarts('c1ccoc1'),                       # Pyran ring (similar to furan)
    'Pyrazine': Chem.MolFromSmarts('c1cnccn1'),                   # Pyrazine ring
    'Pyrrole': Chem.MolFromSmarts('c1cc[nH]c1'),                  # Pyrrole ring
    'Pyridine': Chem.MolFromSmarts('c1ccncc1'),                   # Pyridine ring
    'S-Compounds': Chem.MolFromSmarts('[SX2]'),                   # Sulfur compounds (S with 2 connections)
    'Sulfides': Chem.MolFromSmarts('CSC'),                        # Thioether (sulfide) -C-S-C-
    'Thiazoles': Chem.MolFromSmarts('c1cscn1'),                   # Thiazole ring
    'Nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),                  # Nitro group
    'Sulfonamide': Chem.MolFromSmarts('S(=O)(=O)N'),              # Sulfonamide
    'Thioesters': Chem.MolFromSmarts('C(=O)S'),                   # Thioester
    'Thiols': Chem.MolFromSmarts('[SX2H]'),                       # Thiol group
    'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),                 # Any halogen
    'Tert-butyl': Chem.MolFromSmarts('C(C)(C)C'),                 # Tert-butyl group
    'Nitrile': Chem.MolFromSmarts('C#N')                          # Nitrile group
}

FG_NAMES = list(FG_SMARTS_PATTERNS.keys())

def count_functional_groups(mol):
    counts = []
    for name in FG_NAMES:
        pattern = FG_SMARTS_PATTERNS[name]
        if pattern is None:
            counts.append(0)
            continue
        matches = mol.GetSubstructMatches(pattern)
        counts.append(len(matches))
    return counts