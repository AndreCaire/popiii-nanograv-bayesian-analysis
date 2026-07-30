import numpy as np
import pickle
import os
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.empirical_distr import (EmpiricalDistribution1D,
                                                   EmpiricalDistribution1DKDE,
                                                   EmpiricalDistribution2D,
                                                   EmpiricalDistribution2DKDE)

class JumpProposal(object):
    def __init__(self, signals, empirical_distr=None, save_ext_dists=False, outdir='chains'):
        self.params = []
        self.param_names = []
        self.red_names = []
        self.gw_names = []
        self.empirical_distr = empirical_distr

        for s in signals:
            self.params.extend(s.params)
            self.param_names.extend(s.param_names)
            if s.CP:
                self.gw_names.extend(s.param_names)
            else:
                self.red_names.extend(s.param_names)

        self.pimap = {}
        for ct, p in enumerate(self.param_names):
            self.pimap[p] = ct

        if self.empirical_distr is not None:
            mask = []
            for idx, d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in self.param_names:
                        mask.append(idx)
                else:
                    if (d.param_names[0] in self.param_names and d.param_names[1] in self.param_names):
                        mask.append(idx)
            if len(mask) >= 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
            else:
                self.empirical_distr = None

    def draw_from_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        p = np.random.choice(self.params)
        pidx = self.params.index(p)
        rand = p.sample()
        if type(rand) is np.ndarray:
            subparams = [pn for pn in self.param_names if p.name in pn]
            psubp = np.random.choice(subparams)
            psubidx = subparams.index(psubp)
            pidx = self.param_names.index(psubp)
            rand = rand[psubidx]
        q[pidx] = rand
        lqxy = p.get_logpdf(x[pidx]) - p.get_logpdf(q[pidx])
        return q, float(lqxy)

    def draw_from_red_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        if not self.red_names: return x, 0
        p_name = np.random.choice(self.red_names)
        pidx = self.param_names.index(p_name)
        p = self.params[pidx]
        q[pidx] = p.sample()
        lqxy = p.get_logpdf(x[pidx]) - p.get_logpdf(q[pidx])
        return q, float(lqxy)

    def draw_from_gwb_priors(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        p = np.random.choice(self.params)
        pidx = self.params.index(p)
        rand = p.sample()
        if type(rand) is np.ndarray:
            subparams = [pn for pn in self.param_names if p.name in pn and 'gw' in pn]
            if subparams:
                psubp = np.random.choice(subparams)
                psubidx = subparams.index(psubp)
                pidx = self.param_names.index(psubp)
                rand = rand[psubidx]
        q[pidx] = rand
        lqxy = p.get_logpdf(x[pidx]) - p.get_logpdf(q[pidx])
        return q, float(lqxy)

    def draw_from_empirical_distr(self, x, iter, beta):
        return x, 0

def setup_sampler(ceffyl, outdir, logL, logp, resume=True, jump=True, groups=None, loglkwargs={}, logpkwargs={}, ptmcmc_kwargs={}, empirical_distr=None, save_ext_dists=False):
    if os.path.exists(outdir+'/cov.npy'):
        cov = np.load(outdir+'/cov.npy')
    else:
        cov = np.diag(np.ones(ceffyl.ndim) * 0.1**2)

    if groups is None:
        groups = [list(np.arange(0, ceffyl.ndim))]
        for s in ceffyl.signals:
            groups.append(list(np.hstack(s.pmap)))
            if s.CP:
                [groups.append(list(np.hstack(s.pmap))) for ii in range(10)]

    sampler = ptmcmc(ceffyl.ndim, logL, logp, cov, outDir=outdir, resume=resume, loglkwargs=loglkwargs, logpkwargs=logpkwargs, groups=groups, **ptmcmc_kwargs)

    np.savetxt(outdir+'/pars.txt', ceffyl.param_names, fmt='%s')

    if jump:
        jp = JumpProposal(ceffyl.signals, empirical_distr=empirical_distr, save_ext_dists=save_ext_dists, outdir=outdir)
        sampler.jp = jp
        sampler.addProposalToCycle(jp.draw_from_prior, 5)
        
        red_noise, gw_signal = False, False
        for s in ceffyl.signals:
            if s.CP: gw_signal = True
            else: red_noise = True
        
        if red_noise: sampler.addProposalToCycle(jp.draw_from_red_prior, 10)
        if gw_signal: sampler.addProposalToCycle(jp.draw_from_gwb_priors, 10)

    return sampler
