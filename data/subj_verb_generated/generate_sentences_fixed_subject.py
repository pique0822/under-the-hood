# The sentence format will follow what was proposed by the paper 1 context word before the subject, the subject, 5 words, the verb, and one final word. (W1-S-W5-V-W1).

# The dataset will also be generated with the format,
# A\tS\tline
# A is 1 if the subject and verb agree and 0 otherwise
# S is 1 if the subject is singular and 0 otherwise
# line is the sentence generated as above

# Each subject and verb will be selected from a list noted in this directory as 'singular_subjects.txt', 'plural_subjects.txt', 'singular_verbs.txt', 'plural_verbs.txt'

# The sentence will then be of the form
# 'the ____ that had broken the vase _____ frequently'
base_sentence = 'the {0} that had broken the vase {1} frequently .'
base_sentence = 'the {0} wearing blue and red {1} often .'
base_sentence = 'the dogs with the collar that falls {} often .'

# sing_subj_file = 'subjects/singular_subjects.txt'
# plu_subj_file = 'subjects/plural_subjects.txt'
# sing_verb_file = 'verbs/singular_verbs.txt'
# plu_verb_file = 'verbs/plural_verbs.txt'
verb_file = 'verbs/long_singular_verbs.txt'

out_file = 'datasets/subj_verb_generated_dataset_single_subject.txt'

out_txt = ''

verbs = []

with open(verb_file) as verb_file:
    for verb in verb_file:
        verbs.append(verb.strip())

singular = 1
agreement = 1
for verb in sing_verbs:
    sentence = base_sentence.format(verb.strip())
    out_txt += str(agreement)+'\t'+str(singular)+'\t'+sentence+'\n'

with open(out_file, 'w+') as file:
    file.write(out_txt)
