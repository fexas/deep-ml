import numpy as np

def sequence_logo(seqs):
    # seqs: list of equal-length strings over ACGT
    # return (pwm, info, heights, consensus)
    
    T = len(seqs[0])
    count = np.zeros((T,4))

    for seq in seqs:
        for j in range(T):
            elem = seq[j]
            if elem == 'A':
                count[j,0] += 1
            elif elem == 'C':
                count[j,1] += 1
            elif elem == 'G':
                count[j,2] += 1
            elif elem == 'T':
                count[j,3] += 1
            else:
                 raise ValueError('Value Error')

    pwm = count / len(seqs)
    nlog_pwm = np.where(pwm > 0, -1 * np.log2(pwm), 0)
    ht = np.sum(pwm * nlog_pwm, axis = 1)
    info = 2 - ht
    heights = pwm * info.reshape(-1,1)

    mask = pwm == np.max(pwm,axis=1).reshape(-1,1)
    for i in range(T):
        count1 = 0
        for j in range(4):
            if count1 == 0 and mask[i,j] == 1:
                count1 += 1
            elif count1 != 0 and mask[i,j] == 1:
                mask[i,j] = 0
            else:
                continue

    index = np.arange(4)
    select_index = mask @ index.reshape(-1,1)
    select_index = select_index.flatten()
    slect_index = select_index.astype(int)
    # select_index = np.argmax(pwm, axis = 1)
    select_index = select_index.tolist()
    alphabet = 'ACGT'

    consensus = ''.join(alphabet[id] for id in select_index)

    return pwm, info, heights, consensus


    



