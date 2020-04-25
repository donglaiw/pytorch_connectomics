import sys
import numpy as np
from skimage.measure import label

def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def seg2Count(seg,do_sort=True,rm_zero=False):
    sm = seg.max()
    if sm==0:
        return None,None
    if sm>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        if rm_zero:
            segCounts = segCounts[segIds>0]
            segIds = segIds[segIds>0]
        if do_sort:
            sort_id = np.argsort(-segCounts)
            segIds=segIds[sort_id]
            segCounts=segCounts[sort_id]
    else:
        segIds=np.array([1])
        segCounts=np.array([np.count_nonzero(seg)])
    return segIds, segCounts

def removeSeg(seg, did, invert=False):
    sm = seg.max()
    did = did[did<=sm]
    if invert:
        rl = np.zeros(1+sm).astype(seg.dtype)
        rl[did] = did
    else:
        rl = np.arange(1+sm).astype(seg.dtype)
        rl[did] = 0
    return rl[seg]

def remove_small(seg, thres=100,bid=None):
    if thres>0:
        if bid is None:
            uid, uc = np.unique(seg, return_counts=True)
            bid = uid[uc<thres]
        if len(bid)>0:
            sz = seg.shape
            seg = removeSeg(seg,bid)
    return seg

def relabel(seg, uid=None,nid=None,do_sort=False,do_type=False):
    if seg is None or seg.max()==0:
        return seg
    if do_sort:
        uid,_ = seg2Count(seg,do_sort=True)
    else:
        # get the unique labels
        if uid is None:
            uid = np.unique(seg)
        else:
            uid = np.array(uid)
    uid = uid[uid>0] # leave 0 as 0, the background seg-id
    # get the maximum label for the segment
    mid = int(max(uid)) + 1

    # create an array from original segment id to reduced id
    # format opt
    m_type = seg.dtype
    if do_type:
        mid2 = len(uid) if nid is None else max(nid)+1
        m_type = getSegType(mid2)

    mapping = np.zeros(mid, dtype=m_type)
    if nid is None:
        mapping[uid] = np.arange(1,1+len(uid), dtype=m_type)
    else:
        mapping[uid] = nid.astype(m_type)
    # if uid is given, need to remove bigger seg id 
    seg[seg>=mid] = 0
    return mapping[seg]


def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a[0])==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def label_chunk(get_chunk, numC, rr=1, rm_sz=0, m_type=np.uint64):
    # label chunks or slices
    
    mid = 0
    seg = [None]*numC
    for zi in range(numC):
        print('%d/%d [%d], '%(zi,numC,mid)),
        sys.stdout.flush()
        tmp = get_chunk(zi)>0
        sz = tmp.shape
        numD = len(sz)
        if numD==2:
            tmp = tmp[np.newaxis]

        seg_c = np.zeros(sz).astype(m_type)
        bb=get_bb(tmp)
        print(bb)
        seg_c[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1] = \
                label(tmp[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]).astype(m_type)

        if rm_sz>0:
            # preserve continuous id
            seg_c = remove_small(seg_c, rm_sz)
            seg_c = relabel(seg_c).astype(m_type)

        if zi == 0: # first seg, relabel seg index        
            print('_%d_'%0)
            slice_b = seg_c[-1]
            seg[zi] = seg_c[:,::rr,::rr] # save a low-res one
            mid += seg[zi].max()
            rlA = np.arange(mid+1,dtype=m_type)
        else: # link to previous slice
            slice_t = seg_c[0]            
            slices = label(np.stack([slice_b>0, slice_t>0],axis=0)).astype(m_type)
            # create mapping for seg cur
            lc = np.unique(seg_c);lc=lc[lc>0]
            rl_c = np.zeros(int(lc.max())+1, dtype=int)
            # merge curr seg
            # for 1 pre seg id -> slices id -> cur seg ids
            l0_p = np.unique(slice_b*(slices[0]>0))
            bbs = get_bb_label2d_v2(slice_b,uid=l0_p)[:,1:] 
            #bbs2 = get_bb_label2d_v2(slices[1])
            print('_%d_'%len(l0_p))
            for i,l in enumerate(l0_p):
                bb = bbs[i]
                sid = np.unique(slices[0,bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(slice_b[bb[0]:bb[1]+1,bb[2]:bb[3]+1]==l))
                sid = sid[sid>0]
                # multiple ids
                if len(sid)==1:
                    #bb = bbs2[bbs2[:,0]==sid,1:]
                    #cid = np.unique(slice_t[bb[0]:bb[1]+1,bb[2]:bb[3]+1]*(slices[1,bb[0]:bb[1]+1,bb[2]:bb[3]+1]==sid))
                    cid = np.unique(slice_t*(slices[1]==sid))
                else:
                    cid = np.unique(slice_t*np.in1d(slices[1].reshape(-1),sid).reshape(sz[-2:]))
                rl_c[cid[cid>0]] = l
            
            # new id
            new_num = np.where(rl_c==0)[0][1:] # except the first one
            new_id = np.arange(mid+1,mid+1+len(new_num),dtype=m_type)
            rl_c[new_num] = new_id            
            slice_b = rl_c[seg_c[-1]] # save a high-res
            seg[zi] = rl_c[seg_c[:,::rr,::rr]]
            mid += len(new_num)
            
            # update global id
            rlA = np.hstack([rlA,new_id])
            # merge prev seg
            # for 1 cur seg id -> slices id -> prev seg ids
            l1_c = np.unique(slice_t*(slices[1]>0))
            for l in l1_c:
                sid = np.unique(slices[1]*(slice_t==l))
                sid = sid[sid>0]
                pid = np.unique(slice_b*np.in1d(slices[0].reshape(-1),sid).reshape(sz[-2:]))
                pid = pid[pid>0]
                # get all previous m-to-1 labels
                pid_p = np.where(np.in1d(rlA,rlA[pid]))[0]
                if len(pid_p)>1:
                    rlA[pid_p] = pid.max()
        # memory reduction: each seg
        m2_type = getSegType(seg[zi].max())
        seg[zi] = seg[zi].astype(m2_type)
    # memory reduction: final output
    m2_type = getSegType(rlA.max())
    rlA = rlA.astype(m2_type)
    print('output type:',m2_type)

    return rlA[np.vstack(seg)]



def get_bb_label3d_v2(seg,do_count=False, uid=None):
    sz = seg.shape
    assert len(sz)==3
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    um = int(uid.max())
    out = np.zeros((1+um,7+do_count),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1] = sz[0]
    out[:,3] = sz[1]
    out[:,5] = sz[2]

    # for each slice
    zids = np.where((seg>0).sum(axis=1).sum(axis=1)>0)[0]
    for zid in zids:
        sid = np.unique(seg[zid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,1] = np.minimum(out[sid,1],zid)
        out[sid,2] = np.maximum(out[sid,2],zid)

    # for each row
    rids = np.where((seg>0).sum(axis=0).sum(axis=1)>0)[0]
    for rid in rids:
        sid = np.unique(seg[:,rid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,3] = np.minimum(out[sid,3],rid)
        out[sid,4] = np.maximum(out[sid,4],rid)
    
    # for each col
    cids = np.where((seg>0).sum(axis=0).sum(axis=0)>0)[0]
    for cid in cids:
        sid = np.unique(seg[:,:,cid])
        sid = sid[(sid>0)*(sid<=um)]
        out[sid,5] = np.minimum(out[sid,5],cid)
        out[sid,6] = np.maximum(out[sid,6],cid)

    if do_count:
        ui,uc = np.unique(seg,return_counts=True)
        out[ui[ui<=um],-1]=uc[ui<=um]

    return out[uid]


