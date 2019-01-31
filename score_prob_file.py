# Take a file with probabilities and get score
import argparse
import unicodecsv as csv

parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--prob_csv_path', type=str, default='../data/askubuntu_seg_cand_test_pooja_test_20190129_all_labels_probs-wiki.csv',
                    help='csv containing segments to score')
parser.add_argument('--tag_name', type=str, default='Human_Generated',
                    help='csv containing segments to score')

args = parser.parse_args()

def get_best_f1_score(prob_csv_path, tag_name):
	lab_probs = []
	tag_total = 0
	with open(prob_csv_path) as fr:
		csvr = csv.reader(fr)
		row_num = 0
		for row in csvr:
			row_num += 1
			if row_num == 1:
				continue
			lab_probs.append((row[2], float(row[3])))
			if tag_name == row[2]:
				tag_total += 1

	lab_probs = sorted(lab_probs, key=lambda x: x[1], reverse=True)
	print lab_probs
	tag_labeled = 0
	best_f1 = 0
	for i, (tag, prob) in enumerate(lab_probs):
		if tag == tag_name:
			tag_labeled += 1

		precision = tag_labeled / float(i+1)
		recall = tag_labeled / float(tag_total)

		f1 = 2 * precision * recall / (precision + recall)

		if f1 > best_f1:
			best_f1 = f1

	return best_f1



if __name__ == "__main__":
	#print(get_prob(args.save_dir, "What the hell"))
	best_f1 = get_best_f1_score(args.prob_csv_path, args.tag_name)
	print best_f1