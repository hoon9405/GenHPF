import glob
import logging
import os
from datetime import datetime

import pandas as pd
import pyspark.sql.functions as F
from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("mimiciii")
class MIMICIII(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciii"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(
                    url="https://physionet.org/files/mimiciii/1.4/", dest=self.data_dir
                )

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info("`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet.")
                self.download_ccs_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        # constants
        self._icustay_fname = "ICUSTAYS" + self.ext
        self._patient_fname = "PATIENTS" + self.ext
        self._admission_fname = "ADMISSIONS" + self.ext
        self._diagnosis_fname = "DIAGNOSES_ICD" + self.ext

        # XXX more features? user choice?
        self.tables = [
            {
                "fname": "LABEVENTS" + self.ext,
                "timestamp": "CHARTTIME",
                "timeoffsetunit": "abs",
                "exclude": ["ROW_ID", "SUBJECT_ID"],
                "code": ["ITEMID"],
                "desc": ["D_LABITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "PRESCRIPTIONS" + self.ext,
                "timestamp": "STARTDATE",
                "timeoffsetunit": "abs",
                "exclude": ["ENDDATE", "GSN", "NDC", "ROW_ID", "SUBJECT_ID"],
            },
            {
                "fname": "INPUTEVENTS_MV" + self.ext,
                "timestamp": "STARTTIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "ENDTIME",
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "SUBJECT_ID",
                    "CONTINUEINNEXTDEPT",
                    "CANCELREASON",
                    "STATUSDESCRIPTION",
                    "COMMENTS_CANCELEDBY",
                    "COMMENTS_DATE",
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
            {
                "fname": "INPUTEVENTS_CV" + self.ext,
                "timestamp": "CHARTTIME",
                "timeoffsetunit": "abs",
                "exclude": [
                    "STORETIME",
                    "CGID",
                    "ORDERID",
                    "LINKORDERID",
                    "ROW_ID",
                    "STOPPED",
                    "SUBJECT_ID",
                ],
                "code": ["ITEMID"],
                "desc": ["D_ITEMS" + self.ext],
                "desc_key": ["LABEL"],
            },
        ]

        if self.feature == "select":
            extra_exclude_feature_dict = {
                "LABEVENTS" + self.ext: ["VALUE", "FLAG"],
                "PRESCRIPTIONS"
                + self.ext: [
                    "DRUG_TYPE",
                    "DRUG_NAME_POE",
                    "DRUG_NAME_GENERIC",
                    "FORMULARY_DRUG_CD",
                    "FORM_VAL_DISP",
                    "FORM_UNIT_DISP",
                ],
                "INPUTEVENTS_MV"
                + self.ext: [
                    "AMOUNT",
                    "AMOUNTUOM",
                    "ORDERCATEGORYNAME",
                    "SECONDARYORDERCATEGORYNAME",
                    "ORDERCOMPONENTTYPEDESCRIPTION",
                    "ORDERCATEGORYDESCRIPTION",
                    "PATIENTWEIGHT",
                    "TOTALAMOUNT",
                    "TOTALAMOUNTUOM",
                    "ISOPENBAG",
                    "COMMENTS_EDITEDBY",
                    "ORIGINALAMOUNT",
                    "ORIGINALRATE",
                ],
                "INPUTEVENTS_CV"
                + self.ext: [
                    "AMOUNT",
                    "AMOUNTUOM",
                    "NEWBOTTLE",
                    "ORIGINALAMOUNT",
                    "ORIGINALRATE",
                    "ORIGIANLAMOUNTUOM",
                    "ORIGINALRATE",
                    "ORIGINALRATEUOM",
                    "ORIGINALSITE",
                ],
            }

            for table in self.tables:
                if table["fname"] in extra_exclude_feature_dict.keys():
                    exclude_target_list = extra_exclude_feature_dict[table["fname"]]
                    table["exclude"].extend(exclude_target_list)

        if self.emb_type == "codebase":
            feature_types_for_codebase_emb_dict = {
                "LABEVENTS"
                + self.ext: {
                    "numeric_feat": ["VALUE", "VALUENUM"],
                    "categorical_feat": [],
                    "code_feat": ["ITEMID"],
                },
                "PRESCRIPTIONS"
                + self.ext: {
                    "numeric_feat": ["DOSE_VAL_RX", "FORM_VAL_DISP"],
                    "categorical_feat": [],
                    "code_feat": ["DRUG"],
                },
                "INPUTEVENTS_MV"
                + self.ext: {
                    "numeric_feat": [
                        "AMOUNT",
                        "RATE",
                        "PATIENTWEIGHT",
                        "TOTALAMOUNT",
                        "ORIGINALAMOUNT",
                        "ORIGINALRATE",
                    ],
                    "categorical_feat": ["ISOPENBAG", "CONTINUEINNEXTDEPT", "CANCELREASON", "VALUECOUNTS"],
                    "code_feat": ["ITEMID"],
                },
                "INPUTEVENTS_CV"
                + self.ext: {
                    "numeric_feat": [
                        "AMOUNT",
                        "RATE",
                        "PATIENTWEIGHT",
                        "TOTALAMOUNT",
                        "ORIGINALAMOUNT",
                        "ORIGINALRATE",
                    ],
                    "categorical_feat": ["ISOPENBAG", "CONTINUEINNEXTDEPT", "CANCELREASON", "VALUECOUNTS"],
                    "code_feat": ["ITEMID"],
                },
            }

            for table in self.tables:
                if table["fname"] in feature_types_for_codebase_emb_dict.keys():
                    feature_dict = feature_types_for_codebase_emb_dict[table["fname"]]
                    table.update(feature_dict)

        if self.creatinine or self.bilirubin or self.platelets or self.wbc:
            self.task_itemids = {
                "creatinine": {
                    "fname": "LABEVENTS" + self.ext,
                    "timestamp": "CHARTTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["ROW_ID", "SUBJECT_ID", "VALUE", "VALUEUOM", "FLAG"],
                    "code": ["ITEMID"],
                    "value": ["VALUENUM"],
                    "itemid": [50912],
                },
                "bilirubin": {
                    "fname": "LABEVENTS" + self.ext,
                    "timestamp": "CHARTTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["ROW_ID", "SUBJECT_ID", "VALUE", "VALUEUOM", "FLAG"],
                    "code": ["ITEMID"],
                    "value": ["VALUENUM"],
                    "itemid": [50885],
                },
                "platelets": {
                    "fname": "LABEVENTS" + self.ext,
                    "timestamp": "CHARTTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["ROW_ID", "SUBJECT_ID", "VALUE", "VALUEUOM", "FLAG"],
                    "code": ["ITEMID"],
                    "value": ["VALUENUM"],
                    "itemid": [51265],
                },
                "wbc": {
                    "fname": "LABEVENTS" + self.ext,
                    "timestamp": "CHARTTIME",
                    "timeoffsetunit": "abs",
                    "exclude": ["ROW_ID", "SUBJECT_ID", "VALUE", "FLAG"],
                    "code": ["ITEMID"],
                    "value": ["VALUENUM"],
                    "itemid": [51300, 51301],
                },
                "dialysis": {
                    "tables": {
                        "chartevents": {
                            "fname": "CHARTEVENTS" + self.ext,
                            "timestamp": "CHARTTIME",
                            "timeoffsetunit": "abs",
                            "include": [
                                "ICUSTAY_ID",
                                "SUBJECT_ID",
                                "ITEMID",
                                "VALUE",
                                "VALUENUM",
                                "CHARTTIME",
                                "ERROR",
                            ],
                            "itemid": {
                                "cv_ce": [
                                    152,
                                    148,
                                    149,
                                    146,
                                    147,
                                    151,
                                    150,
                                    7949,
                                    229,
                                    235,
                                    241,
                                    247,
                                    253,
                                    259,
                                    265,
                                    271,
                                    582,
                                    466,
                                    917,
                                    927,
                                    6250,
                                ],
                                "mv_ce": [
                                    226118,
                                    227357,
                                    225725,
                                    226499,
                                    224154,
                                    225810,
                                    227639,
                                    225183,
                                    227438,
                                    224191,
                                    225806,
                                    225807,
                                    228004,
                                    228005,
                                    228006,
                                    224144,
                                    224145,
                                    224149,
                                    224150,
                                    224151,
                                    224152,
                                    224153,
                                    224404,
                                    224406,
                                    226457,
                                    225959,
                                    224135,
                                    224139,
                                    224146,
                                    225323,
                                    225740,
                                    225776,
                                    225951,
                                    225952,
                                    225953,
                                    225954,
                                    225956,
                                    225958,
                                    225961,
                                    225963,
                                    225965,
                                    225976,
                                    225977,
                                    227124,
                                    227290,
                                    227638,
                                    227640,
                                    227753,
                                ],
                            },
                        },
                        "inputevents_cv": {
                            "fname": "INPUTEVENTS_CV" + self.ext,
                            "timestamp": "CHARTTIME",
                            "timeoffsetunit": "abs",
                            "include": ["SUBJECT_ID", "ITEMID", "AMOUNT", "CHARTTIME"],
                            "itemid": {
                                "cv_ie": [
                                    40788,
                                    40907,
                                    41063,
                                    41147,
                                    41307,
                                    41460,
                                    41620,
                                    41711,
                                    41791,
                                    41792,
                                    42562,
                                    43829,
                                    44037,
                                    44188,
                                    44526,
                                    44527,
                                    44584,
                                    44591,
                                    44698,
                                    44927,
                                    44954,
                                    45157,
                                    45268,
                                    45352,
                                    45353,
                                    46012,
                                    46013,
                                    46172,
                                    46173,
                                    46250,
                                    46262,
                                    46292,
                                    46293,
                                    46311,
                                    46389,
                                    46574,
                                    46681,
                                    46720,
                                    46769,
                                    46773,
                                ]
                            },
                        },
                        "outputevents": {
                            "fname": "OUTPUTEVENTS" + self.ext,
                            "timestamp": "CHARTTIME",
                            "timeoffsetunit": "abs",
                            "include": ["SUBJECT_ID", "ITEMID", "VALUE", "CHARTTIME"],
                            "itemid": {
                                "cv_oe": [
                                    40386,
                                    40425,
                                    40426,
                                    40507,
                                    40613,
                                    40624,
                                    40690,
                                    40745,
                                    40789,
                                    40881,
                                    40910,
                                    41016,
                                    41034,
                                    41069,
                                    41112,
                                    41250,
                                    41374,
                                    41417,
                                    41500,
                                    41527,
                                    41623,
                                    41635,
                                    41713,
                                    41750,
                                    41829,
                                    41842,
                                    41897,
                                    42289,
                                    42388,
                                    42464,
                                    42524,
                                    42536,
                                    42868,
                                    42928,
                                    42972,
                                    43016,
                                    43052,
                                    43098,
                                    43115,
                                    43687,
                                    43941,
                                    44027,
                                    44085,
                                    44193,
                                    44199,
                                    44216,
                                    44286,
                                    44567,
                                    44843,
                                    44845,
                                    44857,
                                    44901,
                                    44943,
                                    45479,
                                    45828,
                                    46230,
                                    46232,
                                    46394,
                                    46464,
                                    46712,
                                    46713,
                                    46715,
                                    46741,
                                ],
                            },
                        },
                        "inputevents_mv": {
                            "fname": "INPUTEVENTS_MV" + self.ext,
                            "timestamp": "STARTTIME",
                            "timeoffsetunit": "abs",
                            "include": ["SUBJECT_ID", "ITEMID", "AMOUNT", "STARTTIME"],
                            "itemid": {"mv_ie": [227536, 227525]},
                        },
                        "datetimeevents": {
                            "fname": "DATETIMEEVENTS" + self.ext,
                            "timestamp": "CHARTTIME",
                            "timeoffsetunit": "abs",
                            "include": ["SUBJECT_ID", "ITEMID", "CHARTTIME"],
                            "itemid": {"mv_de": [225318, 225319, 225321, 225322, 225324]},
                        },
                        "procedureevents_mv": {
                            "fname": "PROCEDUREEVENTS_MV" + self.ext,
                            "timestamp": "STARTTIME",
                            "timeoffsetunit": "abs",
                            "include": ["SUBJECT_ID", "ITEMID", "STARTTIME"],
                            "itemid": {
                                "mv_pe": [225441, 225802, 225803, 225805, 224270, 225809, 225955, 225436]
                            },
                        },
                    }
                },
            }

        self.disch_map_dict = {
            "DISC-TRAN CANCER/CHLDRN H": "Other",
            "DISC-TRAN TO FEDERAL HC": "Other",
            "DISCH-TRAN TO PSYCH HOSP": "Other",
            "HOME": "Home",
            "HOME HEALTH CARE": "Home",
            "HOME WITH HOME IV PROVIDR": "Home",
            "HOSPICE-HOME": "Other",
            "HOSPICE-MEDICAL FACILITY": "Other",
            "ICF": "Other",
            "IN_ICU_MORTALITY": "IN_ICU_MORTALITY",
            "LEFT AGAINST MEDICAL ADVI": "Other",
            "LONG TERM CARE HOSPITAL": "Other",
            "OTHER FACILITY": "Other",
            "REHAB/DISTINCT PART HOSP": "Rehabilitation",
            "SHORT TERM HOSPITAL": "Other",
            "SNF": "Skilled Nursing Facility",
            "SNF-MEDICAID ONLY CERTIF": "Skilled Nursing Facility",
            "Death": "Death",
        }
        self._icustay_key = "ICUSTAY_ID"
        self._hadm_key = "HADM_ID"
        self._patient_key = "SUBJECT_ID"

        self._determine_first_icu = "INTIME"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cohorts is None and cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)

        if self.diagnosis:
            logger.info("Start labeling cohorts for diagnosis prediction.")
            # define diagnosis prediction task
            diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

            ccs_dx = pd.read_csv(self.ccs_path)
            ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
            lvl1 = {x: int(y) - 1 for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()}
            diagnoses["diagnosis"] = diagnoses["ICD9_CODE"].map(lvl1)

            diagnoses = diagnoses[(diagnoses["diagnosis"].notnull()) & (diagnoses["diagnosis"] != 14)]
            diagnoses.loc[diagnoses["diagnosis"] >= 14, "diagnosis"] -= 1
            diagnoses = diagnoses.groupby(self.hadm_key)["diagnosis"].agg(lambda x: list(set(x))).to_frame()
            labeled_cohorts = labeled_cohorts.merge(diagnoses, on=self.hadm_key, how="inner")

            # labeled_cohorts = labeled_cohorts.drop(columns=["ICD9_CODE"])

            logger.info(
                "Done preparing diagnosis prediction for the given cohorts, Cohort Numbers: {}".format(
                    len(labeled_cohorts)
                )
            )

            self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")

        if self.bilirubin or self.platelets or self.creatinine or self.wbc:
            logger.info("Start labeling cohorts for clinical task prediction.")

            labeled_cohorts = spark.createDataFrame(labeled_cohorts)

            if self.bilirubin:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "bilirubin", spark)

            if self.platelets:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "platelets", spark)

            if self.creatinine:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "creatinine", spark)

            if self.wbc:
                labeled_cohorts = self.clinical_task(labeled_cohorts, "wbc", spark)
            # labeled_cohorts = labeled_cohorts.toPandas()

            # self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled.clinical_tasks")

            logger.info("Done preparing clinical task prediction for the given cohorts")

        # self.save_to_cache(labeled_cohorts, self.ehr_name + ".cohorts.labeled")
        return labeled_cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays[icustays["FIRST_CAREUNIT"] == icustays["LAST_CAREUNIT"]]

        icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], infer_datetime_format=True, utc=True)

        icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], infer_datetime_format=True, utc=True)
        icustays = icustays.drop(columns=["ROW_ID"])

        # merge icustays with patients to get DOB
        patients["DOB"] = pd.to_datetime(patients["DOB"], infer_datetime_format=True, utc=True)
        patients = patients[patients["SUBJECT_ID"].isin(icustays["SUBJECT_ID"])]
        patients = patients.drop(columns=["ROW_ID"])[["DOB", "SUBJECT_ID"]]
        icustays = icustays.merge(patients, on="SUBJECT_ID", how="left")

        def calculate_age(birth: datetime, now: datetime):
            age = now.year - birth.year
            if now.month < birth.month:
                age -= 1
            elif (now.month == birth.month) and (now.day < birth.day):
                age -= 1

            return age

        icustays["AGE"] = icustays.apply(lambda x: calculate_age(x["DOB"], x["INTIME"]), axis=1)

        # merge with admissions to get discharge information
        icustays = pd.merge(
            icustays.reset_index(drop=True),
            admissions[["HADM_ID", "DISCHARGE_LOCATION", "DEATHTIME", "DISCHTIME"]],
            how="left",
            on="HADM_ID",
        )
        icustays["DISCHARGE_LOCATION"].replace("DEAD/EXPIRED", "Death", inplace=True)

        icustays["DISCHTIME"] = pd.to_datetime(icustays["DISCHTIME"], infer_datetime_format=True, utc=True)

        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["DISCHARGE_LOCATION"] == "Death")
        )
        icustays["DISCHARGE_LOCATION"] = icustays["DISCHARGE_LOCATION"].map(self.disch_map_dict)

        icustays.rename(columns={"DISCHARGE_LOCATION": "HOS_DISCHARGE_LOCATION"}, inplace=True)
        icustays["DISCHTIME"] = (icustays["DISCHTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        icustays["OUTTIME"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        return icustays

    def clinical_task(self, cohorts, task, spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        timeoffsetunit = self.task_itemids[task]["timeoffsetunit"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())

        merge = cohorts.join(table, on=self.hadm_key, how="inner")

        if timeoffsetunit == "abs":
            merge = merge.withColumn(timestamp, F.to_timestamp(timestamp))

        # For Creatinine task, eliminate icus if patient went through dialysis treatment
        # before (obs_size + pred_size) timestamp
        # see https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/rrt.sql
        if task == "creatinine":
            dialysis_tables = self.task_itemids["dialysis"]["tables"]
            chartevents = spark.read.csv(os.path.join(self.data_dir, "CHARTEVENTS" + self.ext), header=True)
            inputevents_cv = spark.read.csv(
                os.path.join(self.data_dir, "INPUTEVENTS_CV" + self.ext), header=True
            )
            outputevents = spark.read.csv(os.path.join(self.data_dir, "OUTPUTEVENTS" + self.ext), header=True)
            inputevents_mv = spark.read.csv(
                os.path.join(self.data_dir, "INPUTEVENTS_MV" + self.ext), header=True
            )
            datetimeevents = spark.read.csv(
                os.path.join(self.data_dir, "DATETIMEEVENTS" + self.ext), header=True
            )
            procedureevents_mv = spark.read.csv(
                os.path.join(self.data_dir, "PROCEDUREEVENTS_MV" + self.ext), header=True
            )
            icustays = spark.read.csv(os.path.join(self.data_dir, "ICUSTAYS" + self.ext), header=True)

            chartevents = chartevents.select(*dialysis_tables["chartevents"]["include"])
            inputevents_cv = inputevents_cv.select(*dialysis_tables["inputevents_cv"]["include"])
            outputevents = outputevents.select(*dialysis_tables["outputevents"]["include"])
            inputevents_mv = inputevents_mv.select(*dialysis_tables["inputevents_mv"]["include"])
            datetimeevents = datetimeevents.select(*dialysis_tables["datetimeevents"]["include"])
            procedureevents_mv = procedureevents_mv.select(*dialysis_tables["procedureevents_mv"]["include"])

            # Filter dialysis related tables with dialysis condition #TODO: check dialysis condition
            cv_ce = (
                chartevents.filter(F.col("ITEMID").isin(dialysis_tables["chartevents"]["itemid"]["cv_ce"]))
                .filter(F.col("VALUE").isNotNull())
                .filter((F.col("ERROR").isNull()) | (F.col("ERROR") == 0))
                .filter(
                    (
                        (F.col("ITEMID").isin([152, 148, 149, 146, 147, 151, 150]))
                        & (F.col("VALUE").isNotNull())
                    )
                    | (
                        (F.col("ITEMID").isin([229, 235, 241, 247, 253, 259, 265, 271]))
                        & (F.col("VALUE") == "Dialysis Line")
                    )
                    | ((F.col("ITEMID") == 466) & (F.col("VALUE") == "Dialysis RN"))
                    | ((F.col("ITEMID") == 927) & (F.col("VALUE") == "Dialysis Solutions"))
                    | ((F.col("ITEMID") == 6250) & (F.col("VALUE") == "dialys"))
                    | (
                        (F.col("ITEMID") == 917)
                        & (
                            F.col("VALUE").isin(
                                [
                                    "+ INITIATE DIALYSIS",
                                    "BLEEDING FROM DIALYSIS CATHETER",
                                    "FAILED DIALYSIS CATH.",
                                    "FEBRILE SYNDROME;DIALYSIS",
                                    "HYPOTENSION WITH HEMODIALYSIS",
                                    "HYPOTENSION.GLOGGED DIALYSIS",
                                    "INFECTED DIALYSIS CATHETER",
                                ]
                            )
                        )
                    )
                    | (
                        (F.col("ITEMID") == 582)
                        & (
                            F.col("VALUE").isin(
                                [
                                    "CAVH Start",
                                    "CAVH D/C",
                                    "CVVHD Start",
                                    "CVVHD D/C",
                                    "Hemodialysis st",
                                    "Hemodialysis end",
                                ]
                            )
                        )
                    )
                )
            )
            icustays = icustays.filter(F.col("DBSOURCE") == "carevue").select(self.icustay_key)
            cv_ce = cv_ce.join(icustays, on=self.icustay_key, how="inner")
            cv_ie = inputevents_cv.filter(
                F.col("ITEMID").isin(dialysis_tables["inputevents_cv"]["itemid"]["cv_ie"])
            ).filter(F.col("AMOUNT") > 0)
            cv_oe = outputevents.filter(
                F.col("ITEMID").isin(dialysis_tables["outputevents"]["itemid"]["cv_oe"])
            ).filter(F.col("VALUE") > 0)
            mv_ce = (
                chartevents.filter(F.col("ITEMID").isin(dialysis_tables["chartevents"]["itemid"]["mv_ce"]))
                .filter(F.col("VALUENUM") > 0)
                .filter((F.col("ERROR").isNull()) | (F.col("ERROR") == 0))
            )
            mv_ie = inputevents_mv.filter(
                F.col("ITEMID").isin(dialysis_tables["inputevents_mv"]["itemid"]["mv_ie"])
            ).filter(F.col("AMOUNT") > 0)
            mv_de = datetimeevents.filter(
                F.col("ITEMID").isin(dialysis_tables["datetimeevents"]["itemid"]["mv_de"])
            )
            mv_pe = procedureevents_mv.filter(
                F.col("ITEMID").isin(dialysis_tables["procedureevents_mv"]["itemid"]["mv_pe"])
            )

            def dialysis_time(table, timecolumn):
                return table.withColumn("_DIALYSIS_TIME", F.to_timestamp(timecolumn)).select(
                    self.patient_key, "_DIALYSIS_TIME"
                )

            cv_ce, cv_ie, cv_oe, mv_ce, mv_ie, mv_de, mv_pe = (
                dialysis_time(cv_ce, "CHARTTIME"),
                dialysis_time(cv_ie, "CHARTTIME"),
                dialysis_time(cv_oe, "CHARTTIME"),
                dialysis_time(mv_ce, "CHARTTIME"),
                dialysis_time(mv_ie, "STARTTIME"),
                dialysis_time(mv_de, "CHARTTIME"),
                dialysis_time(mv_pe, "STARTTIME"),
            )

            dialysis = cv_ce.union(cv_ie).union(cv_oe).union(mv_ce).union(mv_ie).union(mv_de).union(mv_pe)
            dialysis = dialysis.groupBy(self.patient_key).agg(F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME"))
            merge = merge.join(dialysis, on=self.patient_key, how="left")
            merge = merge.filter(F.isnull("_DIALYSIS_TIME") | (F.col("_DIALYSIS_TIME") > F.col(timestamp)))
            merge = merge.drop("_DIALYSIS_TIME")

        if timeoffsetunit == "abs":
            merge = merge.withColumn(
                timestamp, F.round((F.col(timestamp).cast("long") - F.col("INTIME").cast("long")) / 60)
            )

        # Cohort with events within (obs_size + gap_size) - (obs_size + pred_size)
        merge = merge.filter(((self.obs_size + self.gap_size) * 60) <= F.col(timestamp)).filter(
            ((self.obs_size + self.pred_size) * 60) >= F.col(timestamp)
        )

        # Average value of events
        value_agg = merge.groupBy(self.icustay_key).agg(
            F.mean(value).alias("avg_value")
        )  # TODO: mean/min/max?

        # Labeling
        if task == "bilirubin":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 1.2, 0)
                .when((value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1)
                .when((value_agg.avg_value >= 2.0) & (value_agg.avg_value < 6.0), 2)
                .when((value_agg.avg_value >= 6.0) & (value_agg.avg_value < 12.0), 3)
                .when(value_agg.avg_value >= 12.0, 4),
            )
        elif task == "platelets":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value >= 150, 0)
                .when((value_agg.avg_value >= 100) & (value_agg.avg_value < 150), 1)
                .when((value_agg.avg_value >= 50) & (value_agg.avg_value < 100), 2)
                .when((value_agg.avg_value >= 20) & (value_agg.avg_value < 50), 3)
                .when(value_agg.avg_value < 20, 4),
            )

        elif task == "creatinine":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 1.2, 0)
                .when((value_agg.avg_value >= 1.2) & (value_agg.avg_value < 2.0), 1)
                .when((value_agg.avg_value >= 2.0) & (value_agg.avg_value < 3.5), 2)
                .when((value_agg.avg_value >= 3.5) & (value_agg.avg_value < 5), 3)
                .when(value_agg.avg_value >= 5, 4),
            )

        elif task == "wbc":
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 4, 0)
                .when((value_agg.avg_value >= 4) & (value_agg.avg_value <= 12), 1)
                .when((value_agg.avg_value > 12), 2),
            )

        cohorts = cohorts.join(value_agg.select(self.icustay_key, task), on=self.icustay_key, how="left")

        return cohorts

    def infer_data_extension(self) -> str:
        if len(glob.glob(os.path.join(self.data_dir, "*.csv.gz"))) == 26:
            ext = ".csv.gz"
        elif len(glob.glob(os.path.join(self.data_dir, "*.csv"))) == 26:
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
