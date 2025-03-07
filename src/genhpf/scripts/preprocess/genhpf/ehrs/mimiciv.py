import glob
import logging
import os

import pandas as pd
import pyspark.sql.functions as F
from ehrs import EHR, register_ehr

logger = logging.getLogger(__name__)


@register_ehr("mimiciv")
class MIMICIV(EHR):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.ehr_name = "mimiciv"

        if self.data_dir is None:
            self.data_dir = os.path.join(self.cache_dir, self.ehr_name)

            if not os.path.exists(self.data_dir):
                logger.info(
                    "Data is not found so try to download from the internet. "
                    "Note that this is a restricted-access resource. "
                    "Please log in to physionet.org with a credentialed user."
                )
                self.download_ehr_from_url(url="https://physionet.org/files/mimiciv/2.0/", dest=self.data_dir)

        logger.info("Data directory is set to {}".format(self.data_dir))

        if self.ccs_path is None:
            self.ccs_path = os.path.join(self.cache_dir, "ccs_multi_dx_tool_2015.csv")

            if not os.path.exists(self.ccs_path):
                logger.info("`ccs_multi_dx_tool_2015.csv` is not found so try to download from the internet.")
                self.download_ccs_from_url(self.cache_dir)

        if self.gem_path is None:
            self.gem_path = os.path.join(self.cache_dir, "icd10cmtoicd9gem.csv")

            if not os.path.exists(self.gem_path):
                logger.info("`icd10cmtoicd9gem.csv` is not found so try to download from the internet.")
                self.download_icdgem_from_url(self.cache_dir)

        if self.ext is None:
            self.ext = self.infer_data_extension()

        self._icustay_fname = "icu/icustays" + self.ext
        self._patient_fname = "hosp/patients" + self.ext
        self._admission_fname = "hosp/admissions" + self.ext
        self._diagnosis_fname = "hosp/diagnoses_icd" + self.ext

        self.tables = [
            {
                "fname": "hosp/labevents" + self.ext,
                "timestamp": "charttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "labevent_id",
                    "storetime",
                    "subject_id",
                    "specimen_id",
                    "order_provider_id",  # MIMIC-IV-2.2V added this column
                ],
                "code": ["itemid"],
                "desc": ["hosp/d_labitems" + self.ext],
                "desc_key": ["label"],
            },
            {
                "fname": "hosp/prescriptions" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "gsn",
                    "ndc",
                    "subject_id",
                    "pharmacy_id",
                    "poe_id",
                    "poe_seq",
                    "formulary_drug_cd",
                    "stoptime",
                    "order_provider_id",  # MIMIC-IV-2.2V added this column
                ],
            },
            {
                "fname": "icu/inputevents" + self.ext,
                "timestamp": "starttime",
                "timeoffsetunit": "abs",
                "exclude": [
                    "endtime",
                    "storetime",
                    "orderid",
                    "linkorderid",
                    "subject_id",
                    "continueinnextdept",
                    "statusdescription",
                ],
                "code": ["itemid"],
                "desc": ["icu/d_items" + self.ext],
                "desc_key": ["label"],
            },
        ]

        if self.feature == "select":
            extra_exclude_feature_dict = {
                "hosp/labevents"
                + self.ext: ["value", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"],
                "hosp/prescriptions"
                + self.ext: [
                    "drug_type",
                    "form_rx",
                    "form_val_disp",
                    "form_unit_disp",
                    "doses_per_24_hrs",
                    "route",
                ],
                "icu/inputevents"
                + self.ext: [
                    "amount",
                    "amountuom",
                    "ordercategoryname",
                    "secondaryordercategoryname",
                    "ordercomponenttypedescription",
                    "ordercategorydescription",
                    "patientweight",
                    "totalamount",
                    "totalamountuom",
                    "isopenbag",
                    "originalamount",
                    "originalrate",
                ],
            }

            for table in self.tables:
                if table["fname"] in extra_exclude_feature_dict.keys():
                    exclude_target_list = extra_exclude_feature_dict[table["fname"]]
                    table["exclude"].extend(exclude_target_list)

        if self.emb_type == "codebase":
            feature_types_for_codebase_emb_dict = {
                "hosp/labevents"
                + self.ext: {
                    "numeric_feat": ["value", "valuenum", "ref_range_upper", "ref_range_lower"],
                    "categorical_feat": [],
                    "code_feat": ["itemid"],
                },
                "hosp/prescriptions"
                + self.ext: {
                    "numeric_feat": ["dose_val_rx", "form_val_disp"],
                    "categorical_feat": ["doses_per_24_hrs"],
                    "code_feat": ["drug"],
                },
                "icu/inputevents"
                + self.ext: {
                    "numeric_feat": [
                        "amount",
                        "rate",
                        "patientweight",
                        "totalamount",
                        "originalamount",
                        "originalrate",
                    ],
                    "categorical_feat": ["isopenbag", "continueinnextdept", "cancelreason", "valuecounts"],
                    "code_feat": ["itemid"],
                },
            }

            for table in self.tables:
                if table["fname"] in feature_types_for_codebase_emb_dict.keys():
                    feature_dict = feature_types_for_codebase_emb_dict[table["fname"]]
                    table.update(feature_dict)

        if self.creatinine or self.bilirubin or self.platelets or self.wbc:
            self.task_itemids = {
                "creatinine": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50912],
                },
                "bilirubin": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [50885],
                },
                "platelets": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51265],
                },
                "wbc": {
                    "fname": "hosp/labevents" + self.ext,
                    "timestamp": "charttime",
                    "timeoffsetunit": "abs",
                    "exclude": [
                        "labevent_id",
                        "subject_id",
                        "specimen_id",
                        "storetime",
                        "value",
                        "valueuom",
                        "ref_range_lower",
                        "ref_range_upper",
                        "flag",
                        "priority",
                        "comments",
                    ],
                    "code": ["itemid"],
                    "value": ["valuenum"],
                    "itemid": [51300, 51301, 51755],
                },
                "dialysis": {
                    "tables": {
                        "chartevents": {
                            "fname": "icu/chartevents" + self.ext,
                            "timestamp": "charttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "value", "charttime"],
                            "itemid": {
                                "ce": [
                                    226499,
                                    224154,
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
                                    224153,
                                    226457,
                                ]
                            },
                        },
                        "inputevents": {
                            "fname": "icu/inputevents" + self.ext,
                            "timestamp": "starttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "amount", "starttime"],
                            "itemid": {"ie": [227536, 227525]},
                        },
                        "procedureevents": {
                            "fname": "icu/procedureevents" + self.ext,
                            "timestamp": "starttime",
                            "timeoffsetunit": "abs",
                            "include": ["subject_id", "itemid", "value", "starttime"],
                            "itemid": {"pe": [225441, 225802, 225803, 225805, 225809, 225955]},
                        },
                    }
                },
            }

        self.disch_map_dict = {
            "ACUTE HOSPITAL": "Other",
            "AGAINST ADVICE": "Other",
            "ASSISTED LIVING": "Other",
            "CHRONIC/LONG TERM ACUTE CARE": "Other",
            "HEALTHCARE FACILITY": "Other",
            "HOME": "Home",
            "HOME HEALTH CARE": "Home",
            "HOSPICE": "Other",
            "IN_ICU_MORTALITY": "IN_ICU_MORTALITY",
            "OTHER FACILITY": "Other",
            "PSYCH FACILITY": "Other",
            "REHAB": "Rehabilitation",
            "SKILLED NURSING FACILITY": "Skilled Nursing Facility",
            "Death": "Death",
        }

        self._icustay_key = "stay_id"
        self._hadm_key = "hadm_id"
        self._patient_key = "subject_id"

        self._determine_first_icu = "INTIME"

    def build_cohorts(self, cached=False):
        icustays = pd.read_csv(os.path.join(self.data_dir, self.icustay_fname))

        icustays = self.make_compatible(icustays)
        self.icustays = icustays

        cohorts = super().build_cohorts(icustays, cached=cached)

        return cohorts

    def prepare_tasks(self, cohorts, spark, cached=False):
        if cached:
            labeled_cohorts = self.load_from_cache(self.ehr_name + ".cohorts.labeled")
            if labeled_cohorts is not None:
                return labeled_cohorts

        labeled_cohorts = super().prepare_tasks(cohorts, spark, cached)

        if self.diagnosis:
            logger.info("Start labeling cohorts for diagnosis prediction.")

            # define diagnosis prediction task
            diagnoses = pd.read_csv(os.path.join(self.data_dir, self.diagnosis_fname))

            diagnoses = self.icd10toicd9(diagnoses)

            ccs_dx = pd.read_csv(self.ccs_path)
            ccs_dx["'ICD-9-CM CODE'"] = ccs_dx["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_dx["'CCS LVL 1'"] = ccs_dx["'CCS LVL 1'"].str[1:-1]
            lvl1 = {x: int(y) - 1 for _, (x, y) in ccs_dx[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()}

            diagnoses["diagnosis"] = diagnoses["icd_code_converted"].map(lvl1)

            diagnoses = diagnoses[(diagnoses["diagnosis"].notnull()) & (diagnoses["diagnosis"] != 14)]
            diagnoses.loc[diagnoses["diagnosis"] >= 14, "diagnosis"] -= 1
            diagnoses = diagnoses.groupby(self.hadm_key)["diagnosis"].agg(lambda x: list(set(x))).to_frame()

            labeled_cohorts = labeled_cohorts.merge(diagnoses, on=self.hadm_key, how="inner")

            logger.info("Done preparing diagnosis prediction for the given cohorts")

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

        return labeled_cohorts

    def make_compatible(self, icustays):
        patients = pd.read_csv(os.path.join(self.data_dir, self.patient_fname))
        admissions = pd.read_csv(os.path.join(self.data_dir, self.admission_fname))

        # prepare icustays according to the appropriate format
        icustays = icustays.rename(
            columns={
                "los": "LOS",
                "intime": "INTIME",
                "outtime": "OUTTIME",
            }
        )
        admissions = admissions.rename(
            columns={
                "dischtime": "DISCHTIME",
            }
        )

        icustays = icustays[icustays["first_careunit"] == icustays["last_careunit"]]
        icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], infer_datetime_format=True, utc=True)
        icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], infer_datetime_format=True, utc=True)

        icustays = icustays.merge(patients, on="subject_id", how="left")
        icustays["AGE"] = icustays["INTIME"].dt.year - icustays["anchor_year"] + icustays["anchor_age"]

        icustays = icustays.merge(
            admissions[[self.hadm_key, "discharge_location", "deathtime", "DISCHTIME"]],
            how="left",
            on=self.hadm_key,
        )

        icustays["discharge_location"].replace("DIED", "Death", inplace=True)
        icustays["DISCHTIME"] = pd.to_datetime(icustays["DISCHTIME"], infer_datetime_format=True, utc=True)

        icustays["IN_ICU_MORTALITY"] = (
            (icustays["INTIME"] < icustays["DISCHTIME"])
            & (icustays["DISCHTIME"] <= icustays["OUTTIME"])
            & (icustays["discharge_location"] == "Death")
        )
        icustays["discharge_location"] = icustays["discharge_location"].map(self.disch_map_dict)
        icustays.rename(columns={"discharge_location": "HOS_DISCHARGE_LOCATION"}, inplace=True)

        icustays["DISCHTIME"] = (icustays["DISCHTIME"] - icustays["INTIME"]).dt.total_seconds() // 60
        icustays["OUTTIME"] = (icustays["OUTTIME"] - icustays["INTIME"]).dt.total_seconds() // 60

        return icustays

    def icd10toicd9(self, dx):
        gem = pd.read_csv(self.gem_path)
        dx_icd_10 = dx[dx["icd_version"] == 10]["icd_code"]

        unique_elem_no_map = set(dx_icd_10) - set(gem["icd10cm"])

        map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
        map_manual = dict.fromkeys(unique_elem_no_map, "NaN")

        for code_10 in map_manual:
            for i in range(len(code_10), 0, -1):
                tgt_10 = code_10[:i]
                if tgt_10 in gem["icd10cm"]:
                    tgt_9 = gem[gem["icd10cm"].str.contains(tgt_10)]["icd9cm"].mode().iloc[0]
                    map_manual[code_10] = tgt_9
                    break

        def icd_convert(icd_version, icd_code):
            if icd_version == 9:
                return icd_code

            elif icd_code in map_cms:
                return map_cms[icd_code]

            elif icd_code in map_manual:
                return map_manual[icd_code]
            else:
                logger.warn("WRONG CODE: " + icd_code)

        dx["icd_code_converted"] = dx.apply(lambda x: icd_convert(x["icd_version"], x["icd_code"]), axis=1)
        return dx

    def clinical_task(self, cohorts, task, spark):
        fname = self.task_itemids[task]["fname"]
        timestamp = self.task_itemids[task]["timestamp"]
        excludes = self.task_itemids[task]["exclude"]
        code = self.task_itemids[task]["code"][0]
        value = self.task_itemids[task]["value"][0]
        itemid = self.task_itemids[task]["itemid"]

        table = spark.read.csv(os.path.join(self.data_dir, fname), header=True)
        table = table.drop(*excludes)
        table = table.filter(F.col(code).isin(itemid)).filter(F.col(value).isNotNull())

        merge = cohorts.join(table, on=self.hadm_key, how="inner")
        merge = merge.withColumn(timestamp, F.to_timestamp(timestamp))

        # Filter Dialysis at here to use abs timestamp & agg by patient_key
        # For Creatinine task, eliminate icus if patient went through dialysis treatment before
        # (obs_size + pred_size / outtime) timestamp
        # see https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/treatment/rrt.sql
        if task == "creatinine":
            dialysis_tables = self.task_itemids["dialysis"]["tables"]

            chartevents = spark.read.csv(
                os.path.join(self.data_dir, "icu/chartevents" + self.ext), header=True
            )
            inputevents = spark.read.csv(
                os.path.join(self.data_dir, "icu/inputevents" + self.ext), header=True
            )
            procedureevents = spark.read.csv(
                os.path.join(self.data_dir, "icu/procedureevents" + self.ext), header=True
            )

            chartevents = chartevents.select(*dialysis_tables["chartevents"]["include"])
            inputevents = inputevents.select(*dialysis_tables["inputevents"]["include"])
            procedureevents = procedureevents.select(*dialysis_tables["procedureevents"]["include"])

            # Filter dialysis related tables with dialysis condition #TODO: check dialysis condition
            ce = chartevents.filter(
                (
                    ((F.col("itemid") == 225965) & (F.col("value") == "In use"))
                    | (F.col("itemid").isin(dialysis_tables["chartevents"]["itemid"]["ce"]))
                    & F.col("value").isNotNull()
                )
            )
            ie = inputevents.filter(
                F.col("itemid").isin(dialysis_tables["inputevents"]["itemid"]["ie"])
            ).filter(F.col("amount") > 0)
            pe = procedureevents.filter(
                F.col("itemid").isin(dialysis_tables["procedureevents"]["itemid"]["pe"])
            ).filter(F.col("value").isNotNull())

            # Extract Dialysis Times!
            def dialysis_time(table, timecolumn):
                return table.withColumn("_DIALYSIS_TIME", F.to_timestamp(timecolumn)).select(
                    self.patient_key, "_DIALYSIS_TIME"
                )

            ce, ie, pe = (
                dialysis_time(ce, "charttime"),
                dialysis_time(ie, "starttime"),
                dialysis_time(pe, "starttime"),
            )
            dialysis = ce.union(ie).union(pe)
            dialysis = dialysis.groupby(self.patient_key).agg(F.min("_DIALYSIS_TIME").alias("_DIALYSIS_TIME"))
            merge = merge.join(dialysis, on=self.patient_key, how="left")
            # Only leave events with no dialysis / before first dialysis
            merge = merge.filter(F.isnull("_DIALYSIS_TIME") | (F.col("_DIALYSIS_TIME") > F.col(timestamp)))
            merge = merge.drop("_DIALYSIS_TIME")

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
            # NOTE: unit is mg/L
            value_agg = value_agg.withColumn(
                task,
                F.when(value_agg.avg_value < 4, 0)
                .when((value_agg.avg_value >= 4) & (value_agg.avg_value <= 12), 1)
                .when((value_agg.avg_value > 12), 2),
            )

        cohorts = cohorts.join(value_agg.select(self.icustay_key, task), on=self.icustay_key, how="left")

        return cohorts

    def infer_data_extension(self) -> str:
        if (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv.gz"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv.gz"))) == 9
        ):
            ext = ".csv.gz"
        elif (
            len(glob.glob(os.path.join(self.data_dir, "hosp", "*.csv"))) == 22
            or len(glob.glob(os.path.join(self.data_dir, "icu", "*.csv"))) == 9
        ):
            ext = ".csv"
        else:
            raise AssertionError(
                "Provided data directory is not correct. Please check if --data is correct. "
                "--data: {}".format(self.data_dir)
            )

        logger.info("Data extension is set to '{}'".format(ext))

        return ext
