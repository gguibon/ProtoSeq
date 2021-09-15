__author__ = "Gaël Guibon"

import os, sys, getopt, gzip, json

def _createDialogData(dialogue, emotion, act):
    dialogData = list()
    for utterance, label, a in zip(dialogue.split('__eou__'), emotion.strip().split(' '), act.strip().split(' ')):
        datum = dict()
        text = utterance.strip().lower().split(' ')
        if text == ['']: continue
        if text !=  ['']: datum['text'] = text
        raw = utterance.strip()
        if raw != ['']: 
            datum['raw'] = raw
            datum['label'] = int(label)
            datum['act'] = int(a)
        dialogData.append(datum)
    return dialogData

def _getData(dialogues, emotions, acts):
    data = [ _createDialogData(dialogue, emotion, act) for dialogue, emotion, act in zip(dialogues, emotions, acts)]
    data = [ conv for conv in data if len(conv) > 0] # remove empty convs
    print('{} dialogs'.format(len(data)))
    return data

def _getAllUtterances(dialogues, emotions, acts):
    utterances = [ utterance for dialog in _getData(dialogues, emotions, acts) for utterance in dialog ]
    print('{} utterances'.format(len(utterances)))
    return utterances

def _parse(in_dir, out_dir, utteranceMode, allMode):
    split = ''
    if allMode:
        split = 'all'
        dial_dir = os.path.join(in_dir, 'dialogues_text.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act.txt')
    else:
        if in_dir.endswith('train'): split = 'train'
        elif in_dir.endswith('test'): split = 'test'
        elif in_dir.endswith('validation'): split = 'validation'
        else: print("Cannot find directory"); sys.exit()
        dial_dir = os.path.join(in_dir, 'dialogues_{}.txt'.format(split))
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_{}.txt'.format(split))
        act_dir = os.path.join(in_dir, 'dialogues_act_{}.txt'.format(split))
    with open(dial_dir, 'r') as f: dialogues = f.read().split('\n')
    with open(emo_dir, 'r') as f: emotions = f.read().split('\n')
    with open(act_dir, 'r') as f: acts = f.read().split('\n')

    assert len(dialogues) == len(emotions) == len(acts)

    if utteranceMode: 
        data = _getAllUtterances(dialogues, emotions, acts)
        out_dailydialog_dir = os.path.join(out_dir, 'dailydialog_utterances_{}.json'.format(split))
    else: 
        data = _getData(dialogues, emotions, acts)
        out_dailydialog_dir = os.path.join(out_dir, 'dailydialog_{}.json'.format(split))

    with open(out_dailydialog_dir, 'w') as f:
        jsonData = [ json.dumps(d) for d in data]
        f.write('\n'.join(jsonData))


def main(argv):

    in_dir = ''
    out_dir = ''
    utteranceMode = False
    allMode = False


    try:
        opts, args = getopt.getopt(argv,"h:i:o:ua",["in_dir=","out_dir=","utterance_mode=","all_dataset="])
    except getopt.GetoptError:
        print("python3 parser_gg.py -i <in_dir> -o <out_dir> -u <utterance_mode (optional)> -a <all_dataset mode (optional)>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser_gg.py -i <in_dir> -o <out_dir> -u <utterance_mode (optional)> -a <all_dataset mode (optional)>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg
        elif opt in ("-o", "--out_dir"):
            out_dir = arg
        elif opt in ("-u", "--utterances"):
            utteranceMode = True
        elif opt in ("-a", "--all_dataset"):
            allMode = True
            
    print("Input directory : ", in_dir)
    print("Ouptut directory: ", out_dir)
    if utteranceMode: print("utteranceMode ON")
    if allMode: print("allMode ON")

    _parse(in_dir, out_dir, utteranceMode, allMode)

if __name__ == '__main__':
    main(sys.argv[1:])






