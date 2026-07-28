import os
import argparse
from collections import Counter
import csv

import pandas as pd


parser = argparse.ArgumentParser(description='Measure MTERMS sectionizer performance')
parser.add_argument('--input_file', type=str, help='.csv file containing gold and MTERMS output data')
parser.add_argument('--output_dir', type=str, default=None, help='directory to save files containing true positive, false positive, and false negative instances')
parser.add_argument('--csv_output', help='produce .csv output files instead of .txt files', action='store_true')
args = parser.parse_args()

excluded_labels = ['Operative Note Surgical Procedure Section',
                   'Overall prognosis for recovery from this episode OASIS',
                   'Postoperative Diagnosis',
                   'Postprocedure Diagnosis Section',
                   'Preoperative Diagnosis',
                   'Procedure Findings',
                   'Procedures Section',
                   'Admission Date',
                   'Allergies Section',
                   'Anesthesia Section',
                   'Attending Physician Name',
                   'Attending physician name',
                   'Birth date',
                   'Current Imaging Procedure Descriptions',
                   'Date last menstrual period',
                   'Demographic information section',
                   'DICOM Object Catalog Section',
                   'Discharge Diet Section',
                   'Discharge plan',
                   'Document Summary',
                   'Drug-Nutrient Interactions',
                   'Emergency contact information',
                   'Encounters',
                   'Follow-up contact supplemental information',
                   'Health insurance card',
                   'Hospital Discharge Date',
                   'Hospital Discharge Diagnosis Section',
                   'Hospital Discharge Disposition',
                   'Hospital Discharge Instructions',
                   'Hospital Discharge Medication Section',
                   'Hospital Discharge Physical',
                   'Hospital Discharge Studies Summary',
                   'Illness or injury onset date and time',
                   'Immunizations',
                   'Interpreter needed',
                   'Key Images',
                   'Laboratory Test Section',
                   'Medical Equipment',
                   'Medication Section',
                   'Medications Administered Section',
                   'MRN',
                   'Operative Note Fluid Section',
                   'Oral or dental status section',
                   'Patient information - acute-CARE',
                   'Payers',
                   'Plan of Care Section',
                   'Planned Procedure Section',
                   'Prior Imaging Procedure Description',
                   'Procedure Description Section',
                   'Procedure Disposition Section',
                   'Procedure Estimated Blood Loss',
                   'Procedure Implants',
                   'Procedure Indications Section',
                   'Procedure Specimens Taken',
                   'Radiology Study -Recommendations',
                   'Results Section',
                   'RN name',
                   'Specimen-related information panel',
                   'Surgical Drains Section',
                   'Vital Signs Section',
                   'Addendum',                              # ? section labels
                   'Advance Directives Section',
                   'Diagnosis Section',
                   'Discharge Summary',
                   'Family History Section',
                   'Findings Section',
                   'General Status Section',
                   'History of Past Illness Section',
                   # 'History of Present Illness Section',
                   'Hospital Admission Diagnosis Section',
                   # 'Hospital Consultations',
                   'Hospital Course Section',
                   'Interventions',
                   'Number One Section',
                   'Number Two Section',
                   'Number Three Section',
                   'Review of Systems Section',
                   'Medical (General) History',
                   'Section Header']


def load_data(file_name):
    df = pd.read_csv(file_name).astype('str')
    df['section'] = df.section.str.strip()
    df['notecsnid'] = df.notecsnid.str[:-2]
    sectionizer_df = df.iloc[:, :3]
    sectionizer_note_ids = set(list(sectionizer_df['note_id']))
    gold_df = df.iloc[:, 3:]
    gold_df = gold_df.rename(columns={'notecsnid': 'note_id', 'segment_content': 'original_text',
                                      'mterms_normalized': 'section'})
    gold_note_ids = set(list(gold_df['note_id']))
    sectionizer_data = sectionizer_df.to_dict('records')
    sectionizer_data_grouped = {note_id: [record for record in sectionizer_data
                                          if record['note_id'] == note_id and record['section'] not in excluded_labels]
                                for note_id in sectionizer_note_ids}
    gold_data = gold_df.to_dict('records')
    gold_data_grouped = {note_id: [record for record in gold_data
                                   if record['note_id'] == note_id and record['section'] not in excluded_labels]
                         for note_id in gold_note_ids}

    return sectionizer_data_grouped, gold_data_grouped


def prf1(predicted, gold, file_path):
    true_pos = []
    false_pos = []
    false_neg = []
    for note_id, pred_sections in predicted.items():
        if note_id in gold:
            gold_sections = [record['section'] for record in gold[note_id]]
            gold_idx = 0
            for pred_idx, pred_section in enumerate(pred_sections):
                if pred_section['section'] in gold_sections[gold_idx:]:
                    relative_gold_idx = gold_idx + gold_sections[gold_idx:].index(pred_section['section'])
                    match = (pred_section, gold[note_id][relative_gold_idx])
                    true_pos.append(match)
                    if relative_gold_idx > gold_idx:
                        false_neg.extend(gold[note_id][gold_idx:relative_gold_idx])
                    gold_idx = relative_gold_idx + 1
                else:
                    false_pos.append(pred_section)
        else:
            false_pos.extend(pred_sections)

    tp_count = len(true_pos)
    fp_count = len(false_pos)
    fn_count = len(false_neg)
    prec = tp_count / (tp_count + fp_count)
    rec = tp_count / (tp_count + fn_count)
    f1 = 2 * prec * rec / (prec + rec)

    print(f'precision: {prec}\nrecall: {rec}\nf1: {f1}')

    if file_path:
        tp_dist = Counter([match[0]['section'] for match in true_pos])
        fp_dist = Counter([mismatch['section'] for mismatch in false_pos])
        fn_dist = Counter([mismatch['section'] for mismatch in false_neg])
        if args.csv_output:
            with open(os.path.join(file_path, 'tp.csv'), 'w', encoding='utf-8', newline='') as f:
                data_writer = csv.writer(f)
                data_writer.writerow(['note_id', 'original_text_pred', 'original_text_gold', 'section_label'])
                for item_pred, item_gold in true_pos:
                    data_writer.writerow([item_gold['note_id'], item_pred['original_text'], item_gold['original_text'],
                                          item_gold['section']])
            with open(os.path.join(file_path, 'fp.csv'), 'w', encoding='utf-8', newline='') as f:
                data_writer = csv.writer(f)
                data_writer.writerow(['note_id', 'original_text', 'section_label'])
                for item_pred in false_pos:
                    data_writer.writerow([item_pred['note_id'], item_pred['original_text'], item_pred['section']])
            with open(os.path.join(file_path, 'fn.csv'), 'w', encoding='utf-8', newline='') as f:
                data_writer = csv.writer(f)
                data_writer.writerow(['note_id', 'original_text', 'section_label'])
                for item_gold in false_neg:
                    data_writer.writerow([item_gold['note_id'], item_gold['original_text'], item_gold['section']])
        else:
            with open(os.path.join(file_path, 'tp.txt'), 'w', encoding='utf-8') as f:
                f.write('true positives:\n\nlabel distribution:\n')
                for label, count in tp_dist.most_common():
                    f.write(f'{label}: {count}\n')
                f.write('\ninstances:\n')
                for item in true_pos:
                    f.write(f'{item}\n')
            with open(os.path.join(file_path, 'fp.txt'), 'w', encoding='utf-8') as f:
                f.write('false positives:\n\nlabel distribution:\n')
                for label, count in fp_dist.most_common():
                    f.write(f'{label}: {count}\n')
                f.write('\ninstances:\n')
                for item in false_pos:
                    f.write(f'{item}\n')
            with open(os.path.join(file_path, 'fn.txt'), 'w', encoding='utf-8') as f:
                f.write('false negatives:\n\nlabel distribution:\n')
                for label, count in fn_dist.most_common():
                    f.write(f'{label}: {count}\n')
                f.write('\ninstances:\n')
                for item in false_neg:
                    f.write(f'{item}\n')


if __name__ == "__main__":
    sectionizer_data, gold_data = load_data(args.input_file)
    prf1(sectionizer_data, gold_data, args.output_dir)
