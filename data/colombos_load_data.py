import glob
import numpy
import os
import pandas
import sys
import urllib
import zipfile

def load(organism, shared = True, data_dir="data/colombos_data"):
    """
    Downloads gene expression data for the specified ogranism from 
    Colombos website (http://www.colombos.net/) and prepares the 
    data for subsequent analysis. 

    Returns:
      expressions - M by N matrix, where M is the number of contrasts and N is
                    the number of genes. The matrix is/isn't Theano shared
                    if shared is True/False.
      contrasts   - a list of M contrast identifiers.
      genes       - a list of N gene names.
      refannot    - a dictionary mapping contrast identifiers to a set of 
                    reference conditions.
      testannot   - a dictionary mapping contrast identifiers to a set of 
                    test conditions.
    """

    source = "http://www.colombos.net/cws_data/compendium_data"
    zipfname = "%s_compendium_data.zip" %(organism)

    # Download data if necessary.
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.isfile(data_dir + "/%s" %(zipfname)):
        print("Downloading %s data..." %(organism))
        urllib.urlretrieve("%s/%s" %(source, zipfname), data_dir + "/%s" %(zipfname))

        # Extract data.
        fh = zipfile.ZipFile(data_dir + "/%s" %(zipfname))
        fh.extractall(data_dir)
        fh.close()

    # Prepare data for later processing.
    print("Preparing %s data..." %(organism))
    expfname = glob.glob(data_dir + "/colombos_%s_exprdata_*.txt" %(organism))[0]
    refannotfname = glob.glob(data_dir + "/colombos_%s_refannot_*.txt" %(organism))[0]
    testannotfname = glob.glob(data_dir + "/colombos_%s_testannot_*.txt" 
                               %(organism))[0]
        
    df = pandas.read_table(expfname, skiprows = 5, header = 1)
    df = df.fillna(0.0)
    genes = df["Gene name"].values
    expressions = df.iloc[:, 3:len(df.columns)].values
    contrasts = numpy.array(open(expfname, 
                                 "r").readline().strip().split('\t')[1:], 
                            dtype = object)
    lines = open(refannotfname, "r").readlines()
    refannot = {}
    for line in lines[1:]:
        contrast, annot = line.strip().split("\t")
        refannot.setdefault(contrast, set())
        refannot[contrast].add(annot)
    lines = open(testannotfname, "r").readlines()
    testannot = {}
    for line in lines[1:]:
        contrast, annot = line.strip().split("\t")
        testannot.setdefault(contrast, set())
        testannot[contrast].add(annot)

    # Transpose and standardize expressions.
    expressions = expressions.T
    expressions_mean = numpy.mean(expressions, axis = 0)
    expressions_std = numpy.std(expressions, axis = 0)
    expressions = (expressions - expressions_mean) / expressions_std
    expressions = numpy.nan_to_num(expressions)

    # Remove extracted files.
#    os.remove(expfname)
#    os.remove(refannotfname)
#    os.remove(testannotfname)

    if shared:
        import theano
        return theano.shared(numpy.asarray(expressions, 
                                           dtype = theano.config.floatX), 
                             borrow = True), \
                             contrasts, genes, refannot, testannot
    return expressions, contrasts, genes, refannot, testannot

def ecoli(shared):
    """ Escherichia coli (4077 x 4321). """

    return load("ecoli", shared)

def bsubt(shared):
    """ Bacillus subtilis (1259 x 4176). """

    return load("bsubt", shared)

def scoel(shared):
    """ Streptomyces coelicolor (371 x 8239). """
    
    return load("scoel", shared)

def paeru(shared):
    """ Pseudomonas aeruginosa (559 x 5647). """

    return load("paeru", shared)

def mtube(shared):
    """ Mycobacterium tuberculosis (1098 x 4068). """

    return load("mtube", shared)

def hpylo(shared):
    """ Helicobacter pylori (133 x 1616). """

    return load("hpylo", shared)

def meta_sente(shared):
    """ Salmonella enterica (cross-strain) (1066 x 6261). """

    return load("meta_sente", shared)

def sente_lt2(shared):
    """ Salmonella enterica serovar Typhimurium LT2 (172 x 4556). """

    return load("sente_lt2", shared)

def sente_14028s(shared):
    """ Salmonella enterica serovar Typhimurium 14028S (681 x 5416). """

    return load("sente_14028s", shared)

def sente_sl1344(shared):
    """ Salmonella enterica serovar Typhimurium SL1344 (213 x 4655). """

    return load("sente_sl1344", shared)

def smeli_1021(shared):
    """ Sinorhizobium meliloti (424 x 6218). """

    return load("smeli_1021", shared)

def cacet(shared):
    """ Clostridium acetobutylicum (377 x 3778). """

    return load("cacet", shared)

def tther(shared):
    """ Thermus thermophilus (444 x 2173). """

    return load("tther", shared)

def banth(shared):
    """ Bacillus anthracis (66 x 5039). """

    return load("banth", shared)

def bcere(shared):
    """ Bacillus cereus (283 x 5231). """

    return load("bcere", shared)

def bthet(shared):
    """ Bacteroides thetaiotaomicron (333 x 4816). """

    return load("bthet", shared)

def cjeju(shared):
    """ Campylobacter jejuni (152 x 1572). """

    return load("cjeju", shared)

def lrham(shared):
    """ Lactobacillus rhamnosus (79 x 2834). """

    return load("lrham", shared)

def mmari(shared):
    """ Methanococcus maripaludis (364 x 1722). """

    return load("mmari", shared)

def sflex(shared):
    """ Shigella flexneri (35 x 4315). """

    return load("sflex", shared)

def spneu(shared):
    """ Streptococcus pneumoniae (68 x 1914). """

    return load("spneu", shared)

def ypest(shared):
    """ Yersinia pestis (36 x 3979). """

    return load("ypest", shared)

def meta_ally2(shared):
    """ Cross-species analysis (11224 x 31982). """

    return load("meta_ally2", shared)
