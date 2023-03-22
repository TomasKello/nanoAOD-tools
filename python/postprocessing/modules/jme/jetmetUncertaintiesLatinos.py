from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetReCalibrator import JetReCalibrator
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
import ROOT
import math
import os
import sys
import re
import copy
import tarfile
import tempfile
import shutil
import numpy as np
import itertools
ROOT.PyConfig.IgnoreCommandLineOptions = True


class jetmetUncertaintiesProducer(Module):
    def __init__(self,
                 era,
                 globalTag,
                 globalTagProd=None,
                 isData=False, 
                 jesUncertainties=["Total"],
                 jesShifts=["Up","Down"],
                 jerUncertainties=False,
                 jerTag="",   
                 archive=None,
                 jetType="AK4PFchs",
                 jetColl="Jet",
                 metBranchNames=["MET"],
                 applySmearing=False,
                 splitJER=False,
                 onlyJER=[""], 
                 redoJEC=False,
                 applyHEMfix=False,
                 saveMETUncs=['T1', 'T1Smear']
     ):

        # globalTagProd only needs to be defined if METFixEE2017 is to be
        # recorrected, and should be the GT that was used for the production
        # of the nanoAOD files
        self.era = era
        self.isData = isData

        # if set to true, Jet_pt_nom will have JER applied. not to be
        # switched on for data.
        self.undoJER = True
        self.orderByPt = True
        self.applySmearing = applySmearing if not isData else False
        self.applyAtLowPt = False #Latinos treatment only
        self.jerUncertainties = jerUncertainties if not isData else False
        self.splitJER = splitJER
        self.onlyJER = onlyJER
        if self.splitJER:
            if "" in self.onlyJER:  
                self.splitJERIDs = [str(i) for i in range(6)]
            else:
                self.splitJERIDs = self.onlyJER
        else:
            self.splitJERIDs = [""]  # "" ID for the overall JER
        self.metBranchNames = metBranchNames
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        # --------------------------------------------------------------------
        # CV: globalTag and jetType not yet used in the jet smearer, as there
        # is no consistent set of txt files for JES uncertainties and JER scale
        # factors and uncertainties yet
        # --------------------------------------------------------------------

        # re-apply nominal JECs and store new pT
        self.redoJEC = redoJEC

        # JEC sources 
        self.jesUncertainties = jesUncertainties
        self.jesShifts = jesShifts   

        # Calculate and save uncertainties on T1Smear MET if this flag is set
        # to True. Otherwise calculate and save uncertainties on T1 MET
        self.saveMETUncs = saveMETUncs

        # smear jet pT to account for measured difference in JER between data
        # and simulation.
        if jerTag != "":
            self.jerInputFileName = jerTag + "_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = jerTag + "_SF_" + jetType + ".txt"
        else:
            print(
                "WARNING: jerTag is empty!!! This module will soon be " \
                + "deprecated! Please use jetmetHelperRun2 in the future."
            )
            if era == "2016":
                self.jerInputFileName = "Summer16_25nsV1_MC_PtResolution_" + jetType + ".txt"
                self.jerUncertaintyInputFileName = "Summer16_25nsV1_MC_SF_" + jetType + ".txt"
            elif era == "2017" or era == "2018":  # use 2017 JER for 2018 for the time being
                self.jerInputFileName = "Fall17_V3_MC_PtResolution_" + jetType + ".txt"
                self.jerUncertaintyInputFileName = "Fall17_V3_MC_SF_" + jetType + ".txt"
            elif era == "2018" and False:  # jetSmearer not working with 2018 JERs yet
                self.jerInputFileName = "Autumn18_V7_MC_PtResolution_" + jetType + ".txt"
                self.jerUncertaintyInputFileName = "Autumn18_V7_MC_SF_" + jetType + ".txt"

        self.jetSmearer = jetSmearer(globalTag, jetType, self.jerInputFileName,
                                     self.jerUncertaintyInputFileName)

        if "AK4" in jetType:
            self.jetBranchName = jetColl
            self.genJetBranchName = "GenJet"
            self.genSubJetBranchName = None
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.lenVar = "n" + self.jetBranchName

        # read jet energy scale (JES) uncertainties
        # (downloaded from https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC )
        self.jesInputArchivePath = os.environ['CMSSW_BASE'] + \
            "/src/PhysicsTools/NanoAODTools/data/jme/"
        # Text files are now tarred so must extract first into temporary
        # directory (gets deleted during python memory management at
        # script exit)
        self.jesArchive = tarfile.open(
            self.jesInputArchivePath + globalTag +
            ".tgz", "r:gz") if not archive else tarfile.open(
                self.jesInputArchivePath + archive + ".tgz", "r:gz")
        self.jesInputFilePath = tempfile.mkdtemp()
        self.jesArchive.extractall(self.jesInputFilePath)

        # to fully re-calculate type-1 MET the JEC that are currently
        # applied are also needed. IS THAT EVEN CORRECT?

        if len(jesUncertainties) == 1 and jesUncertainties[0] == "Total":
            self.jesUncertaintyInputFileName = globalTag + "_Uncertainty_" + jetType + ".txt"
        elif len(jesUncertainties) > 0 and jesUncertainties[0] == "Merged" and not self.isData:
            self.jesUncertaintyInputFileName = "Regrouped_" + \
                globalTag + "_UncertaintySources_" + jetType + ".txt"
        else:
            self.jesUncertaintyInputFileName = globalTag + \
                "_UncertaintySources_" + jetType + ".txt"

        # read all uncertainty source names from the loaded file
        if len(jesUncertainties) > 0 and jesUncertainties[0] in ["All", "Merged"]:
            with open(self.jesInputFilePath + '/' +
                      self.jesUncertaintyInputFileName) as f:
                lines = f.read().split("\n")
                sources = [
                    x for x in lines if x.startswith("[") and x.endswith("]")
                ]
                sources = [x[1:-1] for x in sources]
                self.jesUncertainties = [ source for source in sources if source in jesUncertainties ] 
        if applyHEMfix:
            self.jesUncertainties.append("HEMIssue")

        # Define jet calibrators if there are JES sources to consider
        #if len(self.jesUncertainties) > 0: 
        # Define the jet recalibrator
        self.jetReCalibrator = JetReCalibrator(
            globalTag,
            jetType,
            True,
            self.jesInputFilePath,
            calculateSeparateCorrections=False,
            calculateType1METCorrection=False)

        # Define the recalibrator for level 1 corrections only
        self.jetReCalibratorL1 = JetReCalibrator(
            globalTag,
            jetType,
            False,
            self.jesInputFilePath,
            calculateSeparateCorrections=True,
            calculateType1METCorrection=False,
            upToLevel=1)

        # Define the recalibrators for GT used in nanoAOD production
        # (only needed to reproduce 2017 v2 MET)
        if globalTagProd:
            self.jetReCalibratorProd = JetReCalibrator(
                globalTagProd,
                jetType,
                True,
                self.jesInputFilePath,
                calculateSeparateCorrections=False,
                calculateType1METCorrection=False)
            self.jetReCalibratorProdL1 = JetReCalibrator(
                globalTagProd,
                jetType,
                False,
                self.jesInputFilePath,
                calculateSeparateCorrections=True,
                calculateType1METCorrection=False,
                upToLevel=1)
        else:
            self.jetReCalibratorProd = False
            self.jetReCalibratorProdL1 = False

        # define energy threshold below which jets are considered as "unclustered energy"
        # cf. JetMETCorrections/Type1MET/python/correctionTermsPfMetType1Type2_cff.py
        self.unclEnThreshold = 15.

        # load libraries for accessing JES scale factors and uncertainties
        # from txt files
        for library in [
                "libCondFormatsJetMETObjects", "libPhysicsToolsNanoAODTools"
        ]:
            if library not in ROOT.gSystem.GetLibraries():
                print("Load Library '%s'" % library.replace("lib", ""))
                ROOT.gSystem.Load(library)

    def getJERsplitID(self, pt, eta):
        if not self.splitJER  and "" in self.splitJERIDs:
            return ""
        if abs(eta) < 1.93:
            return "0"
        elif abs(eta) < 2.5:
            return "1"
        elif abs(eta) < 3:
            if pt < 50:
                return "2"
            else:
                return "3"
        else:
            if pt < 50:
                return "4"
            else:
                return "5"

    def beginJob(self):
        if len(self.jesUncertainties) > 0:
            print("Loading jet energy scale (JES) uncertainties from file '%s'" %
              os.path.join(self.jesInputFilePath,
                           self.jesUncertaintyInputFileName))
            #self.jesUncertainty = ROOT.JetCorrectionUncertainty(os.path.join(self.jesInputFilePath, self.jesUncertaintyInputFileName))

            self.jesUncertainty = {}
            # implementation didn't seem to work for factorized JEC,
            # try again another way
            for jesUncertainty in self.jesUncertainties:
                jesUncertainty_label = jesUncertainty
                if jesUncertainty == "Total" and (
                        len(self.jesUncertainties) == 1 or
                    (len(self.jesUncertainties) == 2
                     and "HEMIssue" in self.jesUncertainties)):
                    jesUncertainty_label = ''
                if jesUncertainty != "HEMIssue":
                    pars = ROOT.JetCorrectorParameters(
                           os.path.join(self.jesInputFilePath,self.jesUncertaintyInputFileName),
                           jesUncertainty_label
                           )
                    self.jesUncertainty[jesUncertainty] = ROOT.JetCorrectionUncertainty(pars)

        if not self.isData:
            self.jetSmearer.beginJob()

    def endJob(self):
        if not self.isData:
            self.jetSmearer.endJob()
        shutil.rmtree(self.jesInputFilePath)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree

        #Define alternative jet branch
        self.altJetBranchName = ""
        brToCheck = ["rawFactor","area","muonSubtrFactor","neEmEF","chEmEF"]
        brToUndoJER = ["Idx_preJER","corr_JER"]     
        self.hasThisBr = {}
        self.hasThisBrJER = {}
        for br in brToCheck:
            self.hasThisBr[br] = False
        for br in brToUndoJER:
            self.hasThisBrJER[br] = False
        self.hasMass = False 
        self.hasIdx  = False
        self.canUndoJER = True
        oBrList = copy.deepcopy(self.out._tree.GetListOfBranches())
        for br in oBrList:
            bname = br.GetName()
            for brKey in brToCheck:
                if bname == self.jetBranchName+"_"+brKey: self.hasThisBr[brKey] = True
            for brKey in brToUndoJER:
                if brKey in bname: self.hasThisBrJER[brKey] = True 
            if bname == self.jetBranchName+"_mass": self.hasMass = True
            if bname == self.jetBranchName+"_jetIdx": self.hasIdx = True
        if (not self.hasMass or any(hasThisBr==False for br,hasThisBr in self.hasThisBr.items())) and self.hasIdx:
            self.altJetBranchName = "Jet"
            if not self.hasMass:  
                print("[WARNING] Using alternative mass branch "+self.altJetBranchName+"_mass")
            for br,hasThisBr in self.hasThisBr.items():
                if not hasThisBr:   
                    print("[WARNING] Using alternative "+br+" branch "+self.altJetBranchName+"_"+br) 
        elif not self.hasIdx:
            if not self.hasMass:
                print("[WARNING] No alternative mass branch was found. Ignoring any mass operations.")
            for br,hasThisBr in self.hasThisBr.items():
                if not hasThisBr:
                    print("[ERROR]   No alternative "+br+" branch was found. Aborting...")
            if any(hasThisBr==False for br,hasThisBr in self.hasThisBr.items()):
                sys.exit(1) 
        if any(hasThisBrJER==False for br,hasThisBrJER in self.hasThisBrJER.items()): 
            self.canUndoJER = False   
        if self.undoJER and not self.canUndoJER:
            print("[ERROR]   Unable to undo JER smearing. Aborting...")
            sys.exit(1)  

        #Initialize new branches
        if self.applySmearing or self.redoJEC:
            self.out.branch("%s_pt_raw" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)
            self.out.branch("%s_pt_nom" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)
            self.out.branch("%s_mass_raw" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)
            self.out.branch("%s_mass_nom" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)
        if self.redoJEC:
            self.out.branch("%s_corr_JEC" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)
        if self.applySmearing:
            self.out.branch("%s_corr_JER" % self.jetBranchName,
                            "F",
                            lenVar=self.lenVar)


        for metBranchName in self.metBranchNames:
            self.out.branch("%s_pt" % metBranchName, "F")
            self.out.branch("%s_phi" % metBranchName, "F")

            if not self.isData and "Puppi" not in metBranchName:
                self.out.branch("%s_T1Smear_pt" % metBranchName, "F")
                self.out.branch("%s_T1Smear_phi" % metBranchName, "F")

        for shift in self.jesShifts:
            if self.jerUncertainties:
                for jerID in self.splitJERIDs:
                    print("Creating new branch "+"%s_pt_JER%s%s" %
                                   (self.jetBranchName, jerID, shift))
                    self.out.branch("%s_pt_JER%s%s" %
                                   (self.jetBranchName, jerID, shift),
                                   "F",
                                   lenVar=self.lenVar)
                    self.out.branch("%s_mass_JER%s%s" %
                                   (self.jetBranchName, jerID, shift),
                                   "F",
                                   lenVar=self.lenVar)
                    self.out.branch("%s_phi_JER%s%s" %
                                   (self.jetBranchName, jerID, shift),
                                   "F",
                                   lenVar=self.lenVar)
                    self.out.branch("%s_eta_JER%s%s" %
                                   (self.jetBranchName, jerID, shift),
                                   "F",
                                   lenVar=self.lenVar)
                    if self.hasIdx:
                        self.out.branch("%s_jetIdx_JER%s%s" %
                                   (self.jetBranchName, jerID, shift),
                                   "F",
                                   lenVar=self.lenVar)
                    for metBranchName in self.metBranchNames:
                        if "Puppi" in metBranchName: continue 
                        if 'T1' in self.saveMETUncs:
                            self.out.branch(
                                "%s_pt_JER%s%s" %
                                (metBranchName, jerID, shift), "F")
                            self.out.branch(
                                "%s_phi_JER%s%s" %
                                (metBranchName, jerID, shift), "F")
                        if 'T1Smear' in self.saveMETUncs:
                            self.out.branch(
                                "%s_T1Smear_pt_JER%s%s" %
                                (metBranchName, jerID, shift), "F")
                            self.out.branch(
                                "%s_T1Smear_phi_JER%s%s" %
                                (metBranchName, jerID, shift), "F")

            for jesUncertainty in self.jesUncertainties:
                self.out.branch(
                    "%s_pt_JES%s%s" %
                    (self.jetBranchName, jesUncertainty, shift),
                    "F",
                    lenVar=self.lenVar)
                self.out.branch(
                    "%s_mass_JES%s%s" %
                    (self.jetBranchName, jesUncertainty, shift),
                    "F",
                    lenVar=self.lenVar)
                self.out.branch(
                    "%s_phi_JES%s%s" %
                    (self.jetBranchName, jesUncertainty, shift),
                    "F",
                    lenVar=self.lenVar)
                self.out.branch(
                    "%s_eta_JES%s%s" %
                    (self.jetBranchName, jesUncertainty, shift),
                    "F",
                    lenVar=self.lenVar)
                if self.hasIdx:
                    self.out.branch(
                        "%s_jetIdx_JES%s%s" %
                        (self.jetBranchName, jesUncertainty, shift),
                        "F",
                        lenVar=self.lenVar) 
                for metBranchName in self.metBranchNames:    
                    if 'T1' in self.saveMETUncs:
                        self.out.branch(
                            "%s_pt_JES%s%s" %
                            (metBranchName, jesUncertainty, shift), "F")
                        self.out.branch(
                            "%s_phi_JES%s%s" %
                            (metBranchName, jesUncertainty, shift), "F")
                    if 'T1Smear' in self.saveMETUncs and "Puppi" not in metBranchName:
                        self.out.branch(
                            "%s_T1Smear_pt_JES%s%s" %
                            (metBranchName, jesUncertainty, shift), "F")
                        self.out.branch(
                            "%s_T1Smear_phi_JES%s%s" %
                            (metBranchName, jesUncertainty, shift), "F")
            for metBranchName in self.metBranchNames:
                self.out.branch(
                    "%s_T1_pt_unclustEn%s" % (metBranchName, shift), "F")
                self.out.branch(
                    "%s_T1_phi_unclustEn%s" % (metBranchName, shift), "F")
                if "Puppi" not in metBranchName:
                    self.out.branch(
                        "%s_T1Smear_pt_unclustEn%s" % (metBranchName, shift), "F")
                    self.out.branch(
                        "%s_T1Smear_phi_unclustEn%s" % (metBranchName, shift), "F")

       
        self.isV5NanoAOD = hasattr(inputTree, self.jetBranchName+"_muonSubtrFactor")
        if not self.isV5NanoAOD and self.altJetBranchName != "":
            self.isV5NanoAOD = hasattr(inputTree, self.altJetBranchName+"_muonSubtrFactor")
        print("nanoAODv5 or higher: " + str(self.isV5NanoAOD))

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail,
        go to next event)"""
        # Undo JER smearing if previously applied
        _jets = Collection(event, self.jetBranchName)
        _nJet = getattr(event,"n"+self.jetBranchName)
        if self.undoJER and self.canUndoJER:
            _jetBranchName = self.jetBranchName[0].lower()+self.jetBranchName[1:]
            newjets = Collection(event, self.jetBranchName)
            for _ijet, _jet in enumerate(_jets):
                origIdx = int(_jet[_jetBranchName+"Idx_preJER"])
                corrJER = float(_jet["corr_JER"])
                orig_jet_pt = float(_jet["pt"])/corrJER
                oBrList = copy.deepcopy(self.out._tree.GetListOfBranches())
                for br in oBrList:
                    bname = br.GetName()
                    _bname = bname.replace(self.jetBranchName+"_","")
                    if self.jetBranchName+"_" in bname and hasattr(_jet, _bname):
                        if bname == self.jetBranchName+"_pt":
                            setattr(newjets[origIdx], _bname, orig_jet_pt)   
                        else: 
                            setattr(newjets[origIdx], _bname, getattr(_jet, _bname))   
            jets = newjets
            nJet = _nJet  
        else: 
            jets = _jets
            nJet = _nJet
        lowPtJets = Collection(event,"CorrT1METJet") if self.isV5NanoAOD else []

        # Fix missing branches in the main jet collection 
        if self.altJetBranchName != "":
            altJets = Collection(event, self.altJetBranchName)
            for ijet,jet in enumerate(jets):
                if not self.hasMass:  
                    jet.mass = altJets[jet.jetIdx].mass
                for br,hasThisBr in self.hasThisBr.items():
                    if not hasThisBr: setattr(jet, br, getattr(altJets[jet.jetIdx], br))

        # to subtract out of the jets for proper type-1 MET corrections
        muons = Collection(event, "Muon")
        if not self.isData:
            genJets = Collection(event, self.genJetBranchName)

        # prepare the low pt jets (they don't have a rawFactor)
        for jet in lowPtJets:
            jet.pt = jet.rawPt
            jet.rawFactor = 0
            jet.mass = 0
            # the following dummy values should be removed once the values
            # are kept in nanoAOD
            jet.neEmEF = 0
            jet.chEmEF = 0

        if not self.isData:
            self.jetSmearer.setSeed(event)

        ############################################################
        # Auxiliary function to match reconstructed jets 
        # to generator level ones.
        # (needed to evaluate JER scale factors and uncertainties)
        ############################################################
        rho = getattr(event, self.rhoBranchName)
        def resolution_matching(jet, genjet):
            '''Helper function to match to gen based on pt difference'''
            params = ROOT.PyJetParametersWrapper()
            params.setJetEta(jet.eta)
            params.setJetPt(jet.pt)
            params.setRho(rho)

            resolution = self.jetSmearer.jer.getResolution(params)

            return abs(jet.pt - genjet.pt) < 3 * resolution * jet.pt

        if not self.isData:
            pairs = matchObjectCollection(jets,
                                          genJets,
                                          dRmax=0.2,
                                          presel=resolution_matching)
            lowPtPairs = matchObjectCollection(lowPtJets,
                                               genJets,
                                               dRmax=0.2,
                                               presel=resolution_matching)
            pairs.update(lowPtPairs)

        ###########################################################  
        # Initialize lists to store jet corrections and variations
        ########################################################### 
        jets_pt_raw = []
        jets_pt_jer = [] #not used
        jets_pt_nom = []
        jets_phi = [] #not affected by varations but we want to store them anyway
        jets_eta = [] #not affected by varations but we want to store them anyway
        jets_jetIdx = [] #not affected by varations but we want to store them anyway

        jets_mass_raw = []
        jets_mass_nom = []

        jets_corr_JEC = []
        jets_corr_JER = []

        jets_pt_jesUp = {}
        jets_pt_jesDown = {}

        jets_mass_jesUp = {}
        jets_mass_jesDown = {}

        jets_pt_jerUp = {}
        jets_pt_jerDown = {}
        jets_mass_jerUp = {}
        jets_mass_jerDown = {}
        for jerID in self.splitJERIDs:
            jets_pt_jerUp[jerID] = []
            jets_pt_jerDown[jerID] = []
            jets_mass_jerUp[jerID] = []
            jets_mass_jerDown[jerID] = []

        for jesUncertainty in self.jesUncertainties:
            jets_pt_jesUp[jesUncertainty] = []
            jets_pt_jesDown[jesUncertainty] = []
            jets_mass_jesUp[jesUncertainty] = []
            jets_mass_jesDown[jesUncertainty] = [] 

        ########################## 
        # Loop over MET objects
        ##########################
        defmet = Object(event, "MET")
        for iMET, metBranchName in enumerate(self.metBranchNames):
            met = Object(event, metBranchName)
            if "Puppi" in metBranchName:
                rawmet = Object(event, "RawPuppiMET")
            else:
                rawmet = Object(event, "RawMET") 

            (t1met_px, t1met_py) = (met.pt * math.cos(met.phi),
                                    met.pt * math.sin(met.phi))
            (def_met_px, def_met_py) = (defmet.pt * math.cos(defmet.phi),
                                        defmet.pt * math.sin(defmet.phi))
            (met_px, met_py) = (rawmet.pt * math.cos(rawmet.phi),
                                rawmet.pt * math.sin(rawmet.phi))
            (met_T1_px, met_T1_py) = (met_px, met_py)
            (met_T1Smear_px, met_T1Smear_py) = (met_px, met_py)

            if 'T1' in self.saveMETUncs:
                (met_T1_px_jerUp, met_T1_px_jerDown, met_T1_py_jerUp,
                 met_T1_py_jerDown) = ({}, {}, {}, {})
            if 'T1Smear' in self.saveMETUncs:
                (met_T1Smear_px_jerUp, met_T1Smear_px_jerDown,
                 met_T1Smear_py_jerUp, met_T1Smear_py_jerDown) = ({}, {}, {}, {})
            for jerID in self.splitJERIDs:
                if 'T1' in self.saveMETUncs:
                    (met_T1_px_jerUp[jerID], met_T1_py_jerUp[jerID]) = (met_px,
                                                                        met_py)
                    (met_T1_px_jerDown[jerID], met_T1_py_jerDown[jerID]) = (met_px,
                                                                            met_py)
                if 'T1Smear' in self.saveMETUncs:
                    (met_T1Smear_px_jerUp[jerID],
                     met_T1Smear_py_jerUp[jerID]) = (met_px, met_py)
                    (met_T1Smear_px_jerDown[jerID],
                     met_T1Smear_py_jerDown[jerID]) = (met_px, met_py)

            if 'T1' in self.saveMETUncs:
                (met_T1_px_jesUp, met_T1_py_jesUp) = ({}, {})
                (met_T1_px_jesDown, met_T1_py_jesDown) = ({}, {})
                for jesUncertainty in self.jesUncertainties:
                    met_T1_px_jesUp[jesUncertainty] = met_px
                    met_T1_py_jesUp[jesUncertainty] = met_py
                    met_T1_px_jesDown[jesUncertainty] = met_px
                    met_T1_py_jesDown[jesUncertainty] = met_py
            if 'T1Smear' in self.saveMETUncs:
                (met_T1Smear_px_jesUp, met_T1Smear_py_jesUp) = ({}, {})
                (met_T1Smear_px_jesDown, met_T1Smear_py_jesDown) = ({}, {})
                for jesUncertainty in self.jesUncertainties:
                    met_T1Smear_px_jesUp[jesUncertainty] = met_px
                    met_T1Smear_py_jesUp[jesUncertainty] = met_py
                    met_T1Smear_px_jesDown[jesUncertainty] = met_px
                    met_T1Smear_py_jesDown[jesUncertainty] = met_py

            # variables needed for re-applying JECs to 2017 v2 MET
            delta_x_T1Jet, delta_y_T1Jet = 0, 0
            delta_x_rawJet, delta_y_rawJet = 0, 0

            #################
            # Loop over jets
            #################
            for iJet, jet in enumerate(itertools.chain(jets, lowPtJets)):
                # jet pt and mass corrections
                jet_pt = jet.pt
                jet_mass = jet.mass
                jet_pt_orig = jet_pt
                rawFactor = jet.rawFactor

                if hasattr(jet, "rawFactor"):
                    jet_rawpt = jet_pt * (1 - jet.rawFactor)
                    jet_rawmass = jet_mass * (1 - jet.rawFactor)
                else:
                    jet_rawpt = -1.0 * jet_pt  # If factor not present factor will be saved as -1
                    jet_rawmass = -1.0 * jet_mass  # If factor not present factor will be saved as -1
                
                (jet_pt, jet_mass) = self.jetReCalibrator.correct(jet, rho) #KELLO: investigate why jet_pt is altered at this step (raw factor is wrong?)
                (jet_pt_l1, jet_mass_l1) = self.jetReCalibratorL1.correct(jet, rho)
                jet.pt = jet_pt
                jet.mass = jet_mass

                # Get the JEC factors
                jec = jet_pt / jet_rawpt
                jecL1 = jet_pt_l1 / jet_rawpt
                if self.jetReCalibratorProd:
                    jecProd = (
                        self.jetReCalibratorProd.correct(jet, rho)[0] / jet_rawpt)
                    jecL1Prod = (
                        self.jetReCalibratorProdL1.correct(jet, rho)[0] / jet_rawpt)
                if not self.isData:
                    genJet = pairs[jet]

                # get the jet for type-1 MET
                newjet = ROOT.TLorentzVector()
                if self.isV5NanoAOD:
                    newjet.SetPtEtaPhiM(
                        jet_pt_orig * (1 - jet.rawFactor) *
                        (1 - jet.muonSubtrFactor), jet.eta, jet.phi, jet.mass)
                    muon_pt = jet_pt_orig * \
                        (1 - jet.rawFactor) * jet.muonSubtrFactor
                else:
                    newjet.SetPtEtaPhiM(jet_pt_orig * (1 - jet.rawFactor), jet.eta,
                                        jet.phi, jet.mass)
                    muon_pt = 0
                    if hasattr(jet, 'muonIdx1'):
                        if jet.muonIdx1 > -1:
                            if muons[jet.muonIdx1].isGlobal:
                                newjet = newjet - muons[jet.muonIdx1].p4()
                                muon_pt += muons[jet.muonIdx1].pt
                        if jet.muonIdx2 > -1:
                            if muons[jet.muonIdx2].isGlobal:
                                newjet = newjet - muons[jet.muonIdx2].p4()
                                muon_pt += muons[jet.muonIdx2].pt

                # set the jet pt to the muon subtracted raw pt
                jet.pt = newjet.Pt()
                jet.rawFactor = 0
                # get the proper jet pts for type-1 MET

                jet_pt_noMuL1L2L3 = jet.pt * jec 
                jet_pt_noMuL1 = jet.pt * jecL1  

                # this step is only needed for v2 MET in 2017 when different JECs
                # are applied compared to the nanoAOD production
                if self.jetReCalibratorProd:
                    jet_pt_noMuProdL1L2L3 = jet.pt * jecProd if jet.pt * \
                        jecProd > self.unclEnThreshold else jet.pt
                    jet_pt_noMuProdL1 = jet.pt * jecL1Prod if jet.pt * \
                        jecProd > self.unclEnThreshold else jet.pt
                else:
                    jet_pt_noMuProdL1L2L3 = jet_pt_noMuL1L2L3
                    jet_pt_noMuProdL1 = jet_pt_noMuL1

                # setting jet back to central values
                jet.pt = jet_pt
                jet.rawFactor = rawFactor

                # evaluate JER scale factors and uncertainties
                # cf. https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution and
                # https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution
                if self.applySmearing or self.jerUncertainties:
                    # Get the smearing factors for MET correction
                    (jet_pt_jerNomVal, jet_pt_jerUpVal,
                     jet_pt_jerDownVal) = self.jetSmearer.getSmearValsPt(
                         jet, genJet, rho)
                    #Latinos treatment only
                    if "UL" in self.era and (not self.applyAtLowPt) and jet.pt<50 and abs(jet.eta)>=2.8 and abs(jet.eta)<=3:
                        ( jet_pt_jerNomVal, jet_pt_jerUpVal, jet_pt_jerDownVal ) = ( 1.0, 1.0, 1.0 )
                else:
                    # if you want to do something with JER in data, please add it here.
                    (jet_pt_jerNomVal, jet_pt_jerUpVal, jet_pt_jerDownVal) = (1.0, 1.0, 1.0)

                # these are the important jet pt values
                jet_pt = jet_pt_orig #KELLO temporary overwrite jet_pt to original value, jet_pt is then used as base for JER u/do variation 
                if (self.undoJER and self.canUndoJER) and (self.applySmearing or self.jerUncertainties):   
                    jet_pt_nom = jet_pt * jet_pt_jerNomVal #KELLO nominal pt is nominally smeared for jets outside of JER pt-eta variation region
                else: 
                    jet_pt_nom = jet_pt 
                jet_pt_L1L2L3 = jet_pt_noMuL1L2L3 + muon_pt
                jet_pt_L1 = jet_pt_noMuL1 + muon_pt

                # not nice, but needed for METv2 in 2017
                jet_pt_prodL1L2L3 = jet_pt_noMuProdL1L2L3 + muon_pt
                jet_pt_prodL1 = jet_pt_noMuProdL1 + muon_pt

                if metBranchName == 'METFixEE2017':
                    # get the delta for removing L1L2L3-L1 corrected jets
                    # (corrected with GT from nanoAOD production!!) in the
                    # EE region from the default MET branch.
                    if jet_pt_prodL1L2L3 > self.unclEnThreshold and 2.65 < abs(
                            jet.eta) < 3.14 and jet_rawpt < 50:
                        delta_x_T1Jet += (jet_pt_prodL1L2L3 - jet_pt_prodL1) * \
                            math.cos(jet.phi) + jet_rawpt * math.cos(jet.phi)
                        delta_y_T1Jet += (jet_pt_prodL1L2L3 - jet_pt_prodL1) * \
                            math.sin(jet.phi) + jet_rawpt * math.sin(jet.phi)

                    # get the delta for removing raw jets in the EE region from
                    # the raw MET
                    if jet_pt_prodL1L2L3 > self.unclEnThreshold and 2.65 < abs(
                            jet.eta) < 3.14 and jet_rawpt < 50:
                        delta_x_rawJet += jet_rawpt * math.cos(jet.phi)
                        delta_y_rawJet += jet_rawpt * math.sin(jet.phi)

                jet_mass_nom = jet_pt_jerNomVal * jet_mass if self.applySmearing else jet_mass
                if jet_mass_nom < 0.0:
                    jet_mass_nom *= -1.0

                # don't store the low pt jets in the Jet_pt_nom branch
                # store central values only once for first MET object
                if iJet < nJet and iMET == 0:
                    jets_pt_raw.append(jet_rawpt)
                    jets_pt_nom.append(jet_pt_nom)
                    jets_mass_raw.append(jet_rawmass)
                    jets_mass_nom.append(jet_mass_nom)
                    jets_corr_JEC.append(jet_pt / jet_rawpt)
                    # can be used to undo JER
                    jets_corr_JER.append(jet_pt_jerNomVal)
                    jets_phi.append(jet.phi)
                    jets_eta.append(jet.eta)
                    if self.hasIdx: jets_jetIdx.append(jet.jetIdx) 

                if not self.isData:
                    #evaluate JER uncertainties
                    if self.jerUncertainties:
                        #KELLO: by default store nominally smeared values
                        jet_pt_jerUp = {
                            jerID: jet_pt_nom
                            for jerID in self.splitJERIDs
                        }
                        jet_pt_jerDown = {
                            jerID: jet_pt_nom
                            for jerID in self.splitJERIDs
                        }
                        jet_mass_jerUp = {
                            jerID: jet_mass_nom
                            for jerID in self.splitJERIDs
                        }
                        jet_mass_jerDown = {
                            jerID: jet_mass_nom
                            for jerID in self.splitJERIDs
                        }
                        thisJERID = self.getJERsplitID(jet_pt_nom, jet.eta)
                        if thisJERID in self.splitJERIDs:  
                            #KELLO: for selected JER ID overwrite with varied value based on jet_pt
                            jet_pt_jerUp[thisJERID] = jet_pt_jerUpVal * jet_pt
                            jet_pt_jerDown[thisJERID] = jet_pt_jerDownVal * jet_pt
                            jet_mass_jerUp[thisJERID] = jet_pt_jerUpVal * jet_mass
                            jet_mass_jerDown[thisJERID] = jet_pt_jerDownVal * jet_mass

                    # don't store the low pt jets in the Jet_pt_nom branch
                    # store JER variations just once per first MET object
                    if iJet < nJet and iMET == 0: 
                        for jerID in self.splitJERIDs:
                            if not self.jerUncertainties: continue
                            jets_pt_jerUp[jerID].append(jet_pt_jerUp[jerID])
                            jets_pt_jerDown[jerID].append(jet_pt_jerDown[jerID])
                            jets_mass_jerUp[jerID].append(jet_mass_jerUp[jerID])
                            jets_mass_jerDown[jerID].append(
                                jet_mass_jerDown[jerID])

                    # evaluate JES uncertainties
                    jet_pt_jesUp = {}
                    jet_pt_jesDown = {}
                    jet_pt_jesUpT1 = {}
                    jet_pt_jesDownT1 = {}

                    jet_mass_jesUp = {}
                    jet_mass_jesDown = {}

                    for jesUncertainty in self.jesUncertainties:
                        # cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties
                        # cf. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
                        if jesUncertainty == "HEMIssue":
                            delta = 1.
                            if iJet < nJet and jet_pt_nom > 15 and jet.jetId & 2 and jet.phi > -1.57 and jet.phi < -0.87:
                                if jet.eta > -2.5 and jet.eta < -1.3:
                                    delta = 0.8
                                elif jet.eta <= -2.5 and jet.eta > -3:
                                    delta = 0.65
                            jet_pt_jesUp[jesUncertainty] = jet_pt_nom
                            jet_pt_jesDown[jesUncertainty] = delta * jet_pt_nom
                            jet_mass_jesUp[jesUncertainty] = jet_mass_nom
                            jet_mass_jesDown[jesUncertainty] = delta * jet_mass_nom

                            jet_pt_jesUpT1[jesUncertainty] = jet_pt_L1L2L3
                            jet_pt_jesDownT1[jesUncertainty] = delta * \
                                jet_pt_L1L2L3

                        else:
                            self.jesUncertainty[jesUncertainty].setJetPt(
                                jet_pt_nom)
                            self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                            delta = self.jesUncertainty[
                                jesUncertainty].getUncertainty(True)
                            jet_pt_jesUp[jesUncertainty] = jet_pt_nom * \
                                (1. + delta)
                            jet_pt_jesDown[jesUncertainty] = jet_pt_nom * \
                                (1. - delta)
                            jet_mass_jesUp[jesUncertainty] = jet_mass_nom * \
                                (1. + delta)
                            jet_mass_jesDown[jesUncertainty] = jet_mass_nom * \
                                (1. - delta)

                            # redo JES variations for T1 MET
                            self.jesUncertainty[jesUncertainty].setJetPt(
                                jet_pt_L1L2L3)
                            self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)
                            delta = self.jesUncertainty[
                                jesUncertainty].getUncertainty(True)
                            jet_pt_jesUpT1[jesUncertainty] = jet_pt_L1L2L3 * \
                                (1. + delta)
                            jet_pt_jesDownT1[jesUncertainty] = jet_pt_L1L2L3 * \
                                (1. - delta)

                        # store JES variation just once for first MET object
                        if iJet < nJet and iMET == 0:
                            jets_pt_jesUp[jesUncertainty].append(
                                jet_pt_jesUp[jesUncertainty])
                            jets_pt_jesDown[jesUncertainty].append(
                                jet_pt_jesDown[jesUncertainty])
                            jets_mass_jesUp[jesUncertainty].append(
                                jet_mass_jesUp[jesUncertainty])
                            jets_mass_jesDown[jesUncertainty].append(
                                jet_mass_jesDown[jesUncertainty])

                # progate JER and JES corrections and uncertainties to MET.
                # do NOT propagate JER to PuppiMET
                # only propagate JECs to MET if the corrected pt without the muon
                # is above the threshold
                if jet_pt_noMuL1L2L3 > self.unclEnThreshold and (jet.neEmEF +
                                                                 jet.chEmEF) < 0.9:
                    # do not re-correct for jets that aren't included in METv2 recipe
                    if not (metBranchName == 'METFixEE2017'
                            and 2.65 < abs(jet.eta) < 3.14 and jet.pt *
                            (1 - jet.rawFactor) < 50):
                        # Translate nominal recalibration to MET object only if flag is true
                        jet_cosPhi = math.cos(jet.phi)
                        jet_sinPhi = math.sin(jet.phi)
                        if self.redoJEC:
                            met_T1_px = met_T1_px - \
                                (jet_pt_L1L2L3 - jet_pt_L1) * jet_cosPhi
                            met_T1_py = met_T1_py - \
                                (jet_pt_L1L2L3 - jet_pt_L1) * jet_sinPhi
                            if not self.isData and self.applySmearing and "Puppi" not in metBranchName:
                                met_T1Smear_px = met_T1Smear_px - \
                                    (jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                     jet_pt_L1) * jet_cosPhi
                                met_T1Smear_py = met_T1Smear_py - \
                                    (jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                     jet_pt_L1) * jet_sinPhi
                        # Variations of T1 MET
                        if 'T1' in self.saveMETUncs:
                            if self.jerUncertainties and "Puppi" not in metBranchName:
                                for jerID in self.splitJERIDs:
                                    # For uncertainties on T1 MET, the up/down
                                    # variations are just the centrally smeared
                                    # MET values
                                    jerUpVal, jerDownVal = jet_pt_jerNomVal, jet_pt_jerNomVal
                                    met_T1_px_jerUp[jerID] = met_T1_px_jerUp[jerID] - \
                                        (jet_pt_L1L2L3 * jerUpVal -
                                         jet_pt_L1) * jet_cosPhi
                                    met_T1_py_jerUp[jerID] = met_T1_py_jerUp[jerID] - \
                                        (jet_pt_L1L2L3 * jerUpVal -
                                         jet_pt_L1) * jet_sinPhi
                                    met_T1_px_jerDown[jerID] = met_T1_px_jerDown[
                                        jerID] - (jet_pt_L1L2L3 * jerDownVal -
                                                  jet_pt_L1) * jet_cosPhi
                                    met_T1_py_jerDown[jerID] = met_T1_py_jerDown[
                                        jerID] - (jet_pt_L1L2L3 * jerDownVal -
                                                  jet_pt_L1) * jet_sinPhi

                            # Calculate JES uncertainties on unsmeared MET
                            for jesUncertainty in self.jesUncertainties:
                                met_T1_px_jesUp[
                                    jesUncertainty] = met_T1_px_jesUp[
                                        jesUncertainty] - (
                                            jet_pt_jesUpT1[jesUncertainty]
                                            - jet_pt_L1) * jet_cosPhi
                                met_T1_py_jesUp[
                                    jesUncertainty] = met_T1_py_jesUp[
                                        jesUncertainty] - (
                                            jet_pt_jesUpT1[jesUncertainty]
                                            - jet_pt_L1) * jet_sinPhi
                                met_T1_px_jesDown[
                                    jesUncertainty] = met_T1_px_jesDown[
                                        jesUncertainty] - (
                                            jet_pt_jesDownT1[jesUncertainty]
                                            - jet_pt_L1) * jet_cosPhi
                                met_T1_py_jesDown[
                                    jesUncertainty] = met_T1_py_jesDown[
                                        jesUncertainty] - (
                                            jet_pt_jesDownT1[jesUncertainty]
                                            - jet_pt_L1) * jet_sinPhi
                        # Variations of T1Smear MET
                        if 'T1Smear' in self.saveMETUncs and "Puppi" not in metBranchName:
                            if self.jerUncertainties:
                                for jerID in self.splitJERIDs:
                                    jerUpVal, jerDownVal = jet_pt_jerNomVal, jet_pt_jerNomVal
                                    if jerID == self.getJERsplitID(
                                            jet_pt_nom, jet.eta):
                                        jerUpVal, jerDownVal = jet_pt_jerUpVal, jet_pt_jerDownVal
                                    met_T1Smear_px_jerUp[
                                        jerID] = met_T1Smear_px_jerUp[jerID] - (
                                            jet_pt_L1L2L3 * jerUpVal -
                                            jet_pt_L1) * jet_cosPhi
                                    met_T1Smear_py_jerUp[
                                        jerID] = met_T1Smear_py_jerUp[jerID] - (
                                            jet_pt_L1L2L3 * jerUpVal -
                                            jet_pt_L1) * jet_sinPhi
                                    met_T1Smear_px_jerDown[
                                        jerID] = met_T1Smear_px_jerDown[jerID] - (
                                            jet_pt_L1L2L3 * jerDownVal -
                                            jet_pt_L1) * jet_cosPhi
                                    met_T1Smear_py_jerDown[
                                        jerID] = met_T1Smear_py_jerDown[jerID] - (
                                            jet_pt_L1L2L3 * jerDownVal -
                                            jet_pt_L1) * jet_sinPhi

                            # Calculate JES uncertainties on smeared MET
                            for jesUncertainty in self.jesUncertainties:
                                jesUp_correction_forT1SmearMET = (
                                    jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                    jet_pt_L1) + (
                                        jet_pt_jesUpT1[jesUncertainty] -
                                        jet_pt_L1L2L3)
                                jesDown_correction_forT1SmearMET = (
                                    jet_pt_L1L2L3 * jet_pt_jerNomVal -
                                    jet_pt_L1) + (
                                        jet_pt_jesDownT1[jesUncertainty] -
                                        jet_pt_L1L2L3)
                                met_T1Smear_px_jesUp[jesUncertainty] = met_T1Smear_px_jesUp[jesUncertainty] - \
                                    jesUp_correction_forT1SmearMET * jet_cosPhi
                                met_T1Smear_py_jesUp[jesUncertainty] = met_T1Smear_py_jesUp[jesUncertainty] - \
                                    jesUp_correction_forT1SmearMET * jet_sinPhi
                                met_T1Smear_px_jesDown[jesUncertainty] = met_T1Smear_px_jesDown[jesUncertainty] - \
                                    jesDown_correction_forT1SmearMET * jet_cosPhi
                                met_T1Smear_py_jesDown[jesUncertainty] = met_T1Smear_py_jesDown[jesUncertainty] - \
                                    jesDown_correction_forT1SmearMET * jet_sinPhi
            ########################  
            # End of loop over jets
            ########################

            ######################################
            # Re-order collections based on pT
            ######################################
            orderJER = {'up' : {}, 'do' : {}}
            orderJES = {'up' : {}, 'do' : {}}
            if self.jerUncertainties:
                for jerID in self.splitJERIDs:
                    for shift in self.jesShifts:
                        if "up" in shift.lower():  
                            if self.orderByPt:
                                orderJER['up'][jerID] = sorted(range(len(jets_pt_jerUp[jerID])), key=jets_pt_jerUp[jerID].__getitem__, reverse=True) 
                            else:
                                orderJER['up'][jerID] = range(len(jets_pt_jerUp[jerID]))  
                        elif "do" in shift.lower():
                             if self.orderByPt:
                                orderJER['do'][jerID] = sorted(range(len(jets_pt_jerDown[jerID])), key=jets_pt_jerDown[jerID].__getitem__, reverse=True)
                             else:
                                orderJER['do'][jerID] = range(len(jets_pt_jerDown[jerID]))     
            for jesUncertainty in self.jesUncertainties:
                for shift in self.jesShifts:
                    if "up" in shift.lower():
                        if self.orderByPt:
                            orderJES['up'][jesUncertainty] = sorted(range(len(jets_pt_jesUp[jesUncertainty])), key=jets_pt_jesUp[jesUncertainty].__getitem__, reverse=True)
                        else:
                            orderJES['up'][jesUncertainty] = range(len(jets_pt_jesUp[jesUncertainty]))
                    if "do" in shift.lower():
                        if self.orderByPt:
                            orderJES['do'][jesUncertainty] = sorted(range(len(jets_pt_jesDown[jesUncertainty])), key=jets_pt_jesDown[jesUncertainty].__getitem__, reverse=True)
                        else:
                            orderJES['do'][jesUncertainty] = range(len(jets_pt_jesDown[jesUncertainty]))
  

            #####################################################
            # Propagate "unclustered energy" uncertainty to MET
            #####################################################
            if metBranchName == 'METFixEE2017':
                # Remove the L1L2L3-L1 corrected jets in the EE region from the
                # default MET branch
                def_met_px += delta_x_T1Jet
                def_met_py += delta_y_T1Jet

                # get unclustered energy part that is removed in the v2 recipe
                met_unclEE_x = def_met_px - t1met_px
                met_unclEE_y = def_met_py - t1met_py

                # finalize the v2 recipe for the rawMET by removing the unclustered
                # part in the EE region
                met_T1_px += delta_x_rawJet - met_unclEE_x
                met_T1_py += delta_y_rawJet - met_unclEE_y

                if not self.isData:
                    # apply v2 recipe correction factor also to JER central value
                    # and variations
                    met_T1Smear_px += delta_x_rawJet - met_unclEE_x
                    met_T1Smear_py += delta_y_rawJet - met_unclEE_y

                if 'T1' in self.saveMETUncs:
                    for jerID in self.splitJERIDs:
                        met_T1_px_jerUp[jerID] += delta_x_rawJet - met_unclEE_x
                        met_T1_py_jerUp[jerID] += delta_y_rawJet - met_unclEE_y
                        met_T1_px_jerDown[jerID] += delta_x_rawJet - met_unclEE_x
                        met_T1_py_jerDown[jerID] += delta_y_rawJet - met_unclEE_y

                    for jesUncertainty in self.jesUncertainties:
                        met_T1_px_jesUp[jesUncertainty] += delta_x_rawJet - \
                            met_unclEE_x
                        met_T1_py_jesUp[jesUncertainty] += delta_y_rawJet - \
                            met_unclEE_y
                        met_T1_px_jesDown[jesUncertainty] += delta_x_rawJet - \
                            met_unclEE_x
                        met_T1_py_jesDown[jesUncertainty] += delta_y_rawJet - \
                            met_unclEE_y

                if 'T1Smear' in self.saveMETUncs:
                    for jerID in self.splitJERIDs:
                        met_T1Smear_px_jerUp[jerID] += delta_x_rawJet - \
                            met_unclEE_x
                        met_T1Smear_py_jerUp[jerID] += delta_y_rawJet - \
                            met_unclEE_y
                        met_T1Smear_px_jerDown[jerID] += delta_x_rawJet - \
                            met_unclEE_x
                        met_T1Smear_py_jerDown[jerID] += delta_y_rawJet - \
                            met_unclEE_y

                    for jesUncertainty in self.jesUncertainties:
                        met_T1Smear_px_jesUp[jesUncertainty] += delta_x_rawJet - \
                            met_unclEE_x
                        met_T1Smear_py_jesUp[jesUncertainty] += delta_y_rawJet - \
                            met_unclEE_y
                        met_T1Smear_px_jesDown[
                            jesUncertainty] += delta_x_rawJet - met_unclEE_x
                        met_T1Smear_py_jesDown[
                            jesUncertainty] += delta_y_rawJet - met_unclEE_y

            if not self.isData and False: #FIXME: this should be relative to flag
                (met_T1_px_unclEnUp, met_T1_py_unclEnUp) = (met_T1_px, met_T1_py)
                (met_T1_px_unclEnDown, met_T1_py_unclEnDown) = (met_T1_px, met_T1_py)
                (met_T1Smear_px_unclEnUp, met_T1Smear_py_unclEnUp) = (met_T1Smear_px, met_T1Smear_py)
                (met_T1Smear_px_unclEnDown, met_T1Smear_py_unclEnDown) = (met_T1Smear_px, met_T1Smear_py)
                met_deltaPx_unclEn = getattr(
                    event, metBranchName + "_MetUnclustEnUpDeltaX")
                met_deltaPy_unclEn = getattr(
                    event, metBranchName + "_MetUnclustEnUpDeltaY")
                met_T1_px_unclEnUp = met_T1_px_unclEnUp + met_deltaPx_unclEn
                met_T1_py_unclEnUp = met_T1_py_unclEnUp + met_deltaPy_unclEn
                met_T1_px_unclEnDown = met_T1_px_unclEnDown - met_deltaPx_unclEn
                met_T1_py_unclEnDown = met_T1_py_unclEnDown - met_deltaPy_unclEn
                met_T1Smear_px_unclEnUp = met_T1Smear_px_unclEnUp + met_deltaPx_unclEn
                met_T1Smear_py_unclEnUp = met_T1Smear_py_unclEnUp + met_deltaPy_unclEn
                met_T1Smear_px_unclEnDown = met_T1Smear_px_unclEnDown - met_deltaPx_unclEn
                met_T1Smear_py_unclEnDown = met_T1Smear_py_unclEnDown - met_deltaPy_unclEn

            ################################
            # Fill all initialized branches
            ################################
 
            #Fill nominal branches and JER/JES variations just once
            if iMET == 0:
                if self.applySmearing or self.redoJEC:
                    self.out.fillBranch("%s_pt_raw" % self.jetBranchName, jets_pt_raw)
                    self.out.fillBranch("%s_pt_nom" % self.jetBranchName, jets_pt_nom)
                    self.out.fillBranch("%s_mass_raw" % self.jetBranchName, jets_mass_raw)
                    self.out.fillBranch("%s_mass_nom" % self.jetBranchName, jets_mass_nom)
                if self.redoJEC:
                    self.out.fillBranch("%s_corr_JEC" % self.jetBranchName, jets_corr_JEC)
                if self.applySmearing: 
                    self.out.fillBranch("%s_corr_JER" % self.jetBranchName, jets_corr_JER)

                if self.jerUncertainties:
                    for jerID in self.splitJERIDs:
                        for shift in self.jesShifts:
                            if "up" in shift.lower():
                                self.out.fillBranch(
                                    "%s_pt_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_pt_jerUp[jerID][idx] for idx in orderJER['up'][jerID]])
                                self.out.fillBranch(
                                    "%s_mass_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_mass_jerUp[jerID][idx] for idx in orderJER['up'][jerID]])
                                self.out.fillBranch(
                                    "%s_phi_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_phi[idx] for idx in orderJER['up'][jerID]])
                                self.out.fillBranch(
                                    "%s_eta_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_eta[idx] for idx in orderJER['up'][jerID]])
                                if self.hasIdx:
                                    self.out.fillBranch(
                                        "%s_jetIdx_JER%s%s" % (self.jetBranchName, jerID, shift),
                                        [jets_jetIdx[idx] for idx in orderJER['up'][jerID]]) 
                            elif "do" in shift.lower():
                                self.out.fillBranch(
                                    "%s_pt_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_pt_jerDown[jerID][idx] for idx in orderJER['do'][jerID]])
                                self.out.fillBranch(
                                    "%s_mass_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_mass_jerDown[jerID][idx] for idx in orderJER['do'][jerID]])
                                self.out.fillBranch(
                                    "%s_phi_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_phi[idx] for idx in orderJER['do'][jerID]])
                                self.out.fillBranch(
                                    "%s_eta_JER%s%s" % (self.jetBranchName, jerID, shift),
                                    [jets_eta[idx] for idx in orderJER['do'][jerID]])
                                if self.hasIdx:
                                    self.out.fillBranch(
                                        "%s_jetIdx_JER%s%s" % (self.jetBranchName, jerID, shift),
                                        [jets_jetIdx[idx] for idx in orderJER['do'][jerID]])

                for jesUncertainty in self.jesUncertainties:
                    for shift in self.jesShifts:
                        if "up" in shift.lower():
                            self.out.fillBranch(
                                "%s_pt_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_pt_jesUp[jesUncertainty][idx] for idx in orderJES['up'][jesUncertainty]])
                            self.out.fillBranch(
                                "%s_mass_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_mass_jesUp[jesUncertainty][idx] for idx in orderJES['up'][jesUncertainty]])
                            self.out.fillBranch(
                                "%s_phi_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_phi[idx] for idx in orderJES['up'][jesUncertainty]])
                            self.out.fillBranch(
                                "%s_eta_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_eta[idx] for idx in orderJES['up'][jesUncertainty]])
                            if self.hasIdx:
                                self.out.fillBranch(
                                    "%s_jetIdx_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                    [jets_jetIdx[idx] for idx in orderJES['up'][jesUncertainty]])   
                        if "do" in shift.lower():  
                            self.out.fillBranch(
                                "%s_pt_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_pt_jesDown[jesUncertainty][idx] for idx in orderJES['do'][jesUncertainty]])
                            self.out.fillBranch(
                                "%s_mass_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_mass_jesDown[jesUncertainty][idx] for idx in orderJES['do'][jesUncertainty]]) 
                            self.out.fillBranch(
                                "%s_phi_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_phi[idx] for idx in orderJES['do'][jesUncertainty]])
                            self.out.fillBranch(
                                "%s_eta_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                [jets_eta[idx] for idx in orderJES['do'][jesUncertainty]])
                            if self.hasIdx:
                                self.out.fillBranch(
                                    "%s_jetIdx_JES%s%s" % (self.jetBranchName, jesUncertainty, shift),
                                    [jets_jetIdx[idx] for idx in orderJES['do'][jesUncertainty]])

            #Fill MET branches per each MET object   
            if self.redoJEC:
                self.out.fillBranch("%s_pt" % metBranchName,
                                    math.sqrt(met_T1_px**2 + met_T1_py**2))
                self.out.fillBranch("%s_phi" % metBranchName,
                                math.atan2(met_T1_py, met_T1_px))

                if not self.isData and self.applySmearing and "Puppi" not in metBranchName:
                    self.out.fillBranch(
                        "%s_T1Smear_pt" % metBranchName,
                        math.sqrt(met_T1Smear_px**2 + met_T1Smear_py**2))
                    self.out.fillBranch("%s_T1Smear_phi" % metBranchName,
                                    math.atan2(met_T1Smear_py, met_T1Smear_px))

            if 'T1' in self.saveMETUncs:
                if self.jerUncertainties and "Puppi" not in metBranchName:
                    for jerID in self.splitJERIDs:
                        for shift in self.jesShifts:
                            if "up" in shift.lower():
                                self.out.fillBranch(
                                    "%s_pt_JER%s%s" % (metBranchName, jerID, shift),
                                    math.sqrt(met_T1_px_jerUp[jerID]**2 +
                                          met_T1_py_jerUp[jerID]**2))
                                self.out.fillBranch(
                                    "%s_phi_JER%s%s" % (metBranchName, jerID, shift),
                                    math.atan2(met_T1_py_jerUp[jerID],
                                               met_T1_px_jerUp[jerID])) 
                            if "do" in shift.lower():
                                self.out.fillBranch(
                                    "%s_pt_JER%s%s" % (metBranchName, jerID, shift),
                                    math.sqrt(met_T1_px_jerDown[jerID]**2 +
                                              met_T1_py_jerDown[jerID]**2))
                                self.out.fillBranch(
                                    "%s_phi_JER%s%s" % (metBranchName, jerID, shift),
                                    math.atan2(met_T1_py_jerDown[jerID],
                                               met_T1_px_jerDown[jerID]))

                for jesUncertainty in self.jesUncertainties: 
                    for shift in self.jesShifts:
                        if "up" in shift.lower():
                            self.out.fillBranch(
                                "%s_pt_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.sqrt(met_T1_px_jesUp[jesUncertainty]**2 +
                                          met_T1_py_jesUp[jesUncertainty]**2))
                            self.out.fillBranch(
                                "%s_phi_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.atan2(met_T1_py_jesUp[jesUncertainty],
                                           met_T1_px_jesUp[jesUncertainty]))
                        if "do" in shift.lower():  
                            self.out.fillBranch(
                                "%s_pt_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.sqrt(met_T1_px_jesDown[jesUncertainty]**2 +
                                          met_T1_py_jesDown[jesUncertainty]**2))
                            self.out.fillBranch(
                                "%s_phi_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.atan2(met_T1_py_jesDown[jesUncertainty],
                                           met_T1_px_jesDown[jesUncertainty]))

            if 'T1Smear' in self.saveMETUncs and "Puppi" not in metBranchName:
                if self.jerUncertainties:
                    for jerID in self.splitJERIDs:
                        for shift in self.jesShifts:
                            if "up" in shift.lower():
                                self.out.fillBranch(
                                    "%s_T1Smear_pt_JER%s%s" % (metBranchName, jerID, shift),
                                    math.sqrt(met_T1Smear_px_jerUp[jerID]**2 +
                                              met_T1Smear_py_jerUp[jerID]**2))
                                self.out.fillBranch(
                                    "%s_T1Smear_phi_JER%s%s" % (metBranchName, jerID, shift),
                                    math.atan2(met_T1Smear_py_jerUp[jerID],
                                               met_T1Smear_px_jerUp[jerID]))
                            if "do" in shift.lower():  
                                self.out.fillBranch(
                                    "%s_T1Smear_pt_JER%s%s" %
                                    (metBranchName, jerID, shift),
                                    math.sqrt(met_T1Smear_px_jerDown[jerID]**2 +
                                              met_T1Smear_py_jerDown[jerID]**2))
                                self.out.fillBranch(
                                    "%s_T1Smear_phi_JER%s%s" %
                                    (metBranchName, jerID, shift),
                                    math.atan2(met_T1Smear_py_jerDown[jerID],
                                               met_T1Smear_px_jerDown[jerID]))

                for jesUncertainty in self.jesUncertainties:
                    for shift in self.jesShifts:
                        if "up" in shift.lower():  
                            self.out.fillBranch(
                                "%s_T1Smear_pt_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.sqrt(met_T1Smear_px_jesUp[jesUncertainty]**2 +
                                          met_T1Smear_py_jesUp[jesUncertainty]**2))
                            self.out.fillBranch(
                                "%s_T1Smear_phi_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.atan2(met_T1Smear_py_jesUp[jesUncertainty],
                                           met_T1Smear_px_jesUp[jesUncertainty]))
                        if "do" in shift.lower():  
                            self.out.fillBranch(
                                "%s_T1Smear_pt_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.sqrt(met_T1Smear_px_jesDown[jesUncertainty]**2 +
                                          met_T1Smear_py_jesDown[jesUncertainty]**2))
                            self.out.fillBranch(
                                "%s_T1Smear_phi_JES%s%s" %
                                (metBranchName, jesUncertainty, shift),
                                math.atan2(met_T1Smear_py_jesDown[jesUncertainty],
                                           met_T1Smear_px_jesDown[jesUncertainty]))

            if False: #FIXME
                for shift in self.jesShifts:
                    if "up" in shift.lower():
                        self.out.fillBranch(
                            "%s_T1_pt_unclustEn%s" % (self.metBranchName, shift),
                            math.sqrt(met_T1_px_unclEnUp**2 + met_T1_py_unclEnUp**2))
                        self.out.fillBranch("%s_T1_phi_unclustEn%s" % (self.metBranchName, shift),
                                            math.atan2(met_T1_py_unclEnUp, met_T1_px_unclEnUp))
                        self.out.fillBranch(
                            "%s_T1Smear_pt_unclustEn%s" % (self.metBranchName, shift),
                            math.sqrt(met_T1Smear_px_unclEnUp**2 + met_T1Smear_py_unclEnUp**2))
                        self.out.fillBranch("%s_T1Smear_phi_unclustEn%s" % (self.metBranchName, shift),
                                            math.atan2(met_T1Smear_py_unclEnUp, met_T1Smear_px_unclEnUp))
                    if "do" in shift.lower():   
                        self.out.fillBranch(
                            "%s_T1_pt_unclustEn%s" % (self.metBranchName, shift),
                            math.sqrt(met_T1_px_unclEnDown**2 + met_T1_py_unclEnDown**2))
                        self.out.fillBranch(
                            "%s_T1_phi_unclustEn%s" % (self.metBranchName, shift),
                            math.atan2(met_T1_py_unclEnDown, met_T1_px_unclEnDown))
                        self.out.fillBranch(
                            "%s_T1Smear_pt_unclustEn%s" % (self.metBranchName, shift),
                            math.sqrt(met_T1Smear_px_unclEnDown**2 + met_T1Smear_py_unclEnDown**2))
                        self.out.fillBranch(
                            "%s_T1Smear_phi_unclustEn%s" % (self.metBranchName, shift),
                            math.atan2(met_T1Smear_py_unclEnDown, met_T1Smear_px_unclEnDown))

            ###############################
            # End of loop over MET objects   
            ###############################

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid
# having them loaded when not needed
'''
jetmetUncertainties2016 = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"])
jetmetUncertainties2016All = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"])

jetmetUncertainties2017 = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"])
jetmetUncertainties2017METv2 = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", metBranchName='METFixEE2017')
jetmetUncertainties2017All = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"])

jetmetUncertainties2018 = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"])
jetmetUncertainties2018Data = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_RunB_V8_DATA", archive="Autumn18_V8_DATA", isData=True)
jetmetUncertainties2018All = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"])

jetmetUncertainties2016AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2016AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK4PFPuppi")

jetmetUncertainties2017AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2017AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK4PFPuppi")

jetmetUncertainties2018AK4Puppi = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"], jetType="AK4PFPuppi")
jetmetUncertainties2018AK4PuppiAll = lambda: jetmetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"], jetType="AK4PFPuppi")
'''
