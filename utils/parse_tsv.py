# parse tsv files for their target proteins. 
class PatentTSV:
    def __init__(self, name):
        self.name = name
        self.file = open(self.name)
        self.targets = self.parse_header('Target Name Assigned by Curator or DataSource')
        self.smiles = self.parse_header('Ligand SMILES')
        self.inchi = self.parse_header('Ligand InChI')
        self.file.close()

    def parse_header(self, header_name):
        self.file.seek(0)
        target_col = -1
        header = self.file.readline()
        for i, val in enumerate(header.split('\t')):
            if header_name == val:
                target_col = i

        if target_col == -1:
            return 'Compound not found'

        targets = []
        for line in self.file:
            target = line.split('\t')[target_col:target_col+1]
            if len(target) > 0 and target[0] not in targets:
                targets.append(target[0])
        return targets
