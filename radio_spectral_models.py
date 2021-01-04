# !/bin/python
# Code for all the models used to fit radio sources, paricularly peaked-spectrum sources (GPS/CSS models). All are fitting in linear space.
# Author: J. R. Callingham

import scipy.special as special
import numpy as np

def curve(freq, freq_peak, peakfreq, alphathick, alphathin): # Model taken from Tschager et al. 2003. General fit not based on any physics.
	return freq_peak/(1 -np.exp(-1))*((freq/freq_peak)**alphathick)*(1 - np.exp(-(freq/freq_peak)**(alphathin-alphathick)))

def powlaw(freq,a,alpha): # defining powlaw as S = a*nu^-alpha. Important to have x value first in definition of function.
	return a*(freq**(-alpha))

# Following models are from Tingay and De Kool 2003. 

def singhomobremss(freq,S_norm,alpha,freq_peak): # nonthermal power law spectrum absorbed by a homogeneous free-free absorbing screen.
	return S_norm*(freq**(-alpha))*np.exp(-(freq/freq_peak)**(-2.1))

def doubhomobremss(freq,S_norm1,S_norm2,alpha1,alpha2,freq_peak1,freq_peak2): # Two seperate nonthermal power-law components, each with its own homogeneous free-free absorbing screen.
 	return S_norm1*(freq**(-alpha1))*np.exp(-(freq/freq_peak1)**(-2.1)) + S_norm2*(freq**(-alpha2))*np.exp(-(freq/freq_peak2)**(-2.1))

def doubhomobremsscurve(freq,S_norm1,S_norm2,alpha1,alpha2,freq_peak1,freq_peak2,gamma1,gamma2): # Two seperate nonthermal power-law components, each with its own homogeneous free-free absorbing screen.
 	return S_norm1*(freq**(-alpha1))*np.exp(gamma1*np.power(np.log(freq),2))*np.exp(-(freq/freq_peak1)**(-2.1)) + S_norm2*(freq**(-alpha2))*np.exp(gamma2*np.power(np.log(freq),2))*np.exp(-(freq/freq_peak2)**(-2.1))

def singinhomobremss(freq,S_norm,alpha,p,freq_peak): # Single inhomogeneous free-free emission model	
	return S_norm*(p+1)*((freq/freq_peak)**(2.1*(p+1)-alpha))*special.gammainc((p+1),((freq/freq_peak)**(-2.1)))*special.gamma(p+1)

def doubinhomobremss(freq,S_norm1,S_norm2,alpha1,alpha2,p1,p2,freq_peak1,freq_peak2): # Double inhomogeneous free-free emission model
	return S_norm1*(p1+1)*((freq/freq_peak1)**(2.1*(p1+1)-alpha1))*special.gammainc((p1+1),(freq/freq_peak1)**(-2.1)) + S_norm2*(p2+1)*((freq/freq_peak2)**(2.1*(p2+1)-alpha2))*special.gammainc((p2+1),(freq/freq_peak2)**(-2.1))

def internalbremss(freq,S_norm,alpha,freq_peak): # Internal free-free model
	return S_norm*(freq**(-alpha))*(((1-np.exp(-(freq/freq_peak)**(-2.1)))/((freq/freq_peak)**(-2.1))))

def singSSA(freq,S_norm,beta,peak_freq): # Single SSA model
	return S_norm*((freq/peak_freq)**(-(beta-1)/2))*(1-np.exp(-(freq/peak_freq)**(-(beta+4)/2)))/((freq/peak_freq)**(-(beta+4)/2))

def doubSSA(freq,S_norm1,S_norm2,beta1,beta2,peak_freq1,peak_freq2): # Double SSA model
	return S_norm1*((freq/peak_freq1)**(-(beta1-1)/2))*(1-np.exp(-(freq/peak_freq1)**(-(beta1+4)/2)))/((freq/peak_freq1)**(-(beta1+4)/2)) + S_norm2*((freq/peak_freq2)**(-(beta2-1)/2))*(1-np.exp(-(freq/peak_freq2)**(-(beta2+4)/2)))/((freq/peak_freq2)**(-(beta2+4)/2))

# Following models add curvature to the spectrum. q is the curvature parameter.

def singhomobremsscurve(freq,S_norm,alpha,freq_peak,q):
	return S_norm*(freq**(-alpha))*np.exp(q*np.power(np.log(freq),2))*np.exp(-(freq/freq_peak)**(-2.1))

def curvepowlaw(freq, S_norm, alpha, q):
	return S_norm*(freq**(-alpha))*np.exp(q*np.power(np.log(freq),2))

def singinhomobremsscurve(freq,S_norm,alpha,p,freq_peak,q): # Single inhomogeneous free-free emission model
	return S_norm*(p+1)*(np.power((freq/freq_peak),(2.1*(p+1)))*np.power(freq,-alpha)*np.power(freq_peak,alpha)*np.exp(q*np.power(np.log(freq),2))*special.gammainc((p+1),(freq/freq_peak)**(-2.1)))*special.gamma(p+1)

def doubhomobremsscurve(freq,S_norm1,S_norm2,alpha1,alpha2,freq_peak1,freq_peak2,q1,q2): 
 	return S_norm1*(freq**(-alpha1))*np.exp(q1*np.power(np.log(freq),2))*np.exp(-(freq/freq_peak1)**(-2.1)) + S_norm2*(freq**(-alpha2))*np.exp(q2*np.power(np.log(freq),2))*np.exp(-(freq/freq_peak2)**(-2.1))

def duffcurve(freq,S_norm,q,peak_freq): # Duffy & Blundell 2012 curved model
	return S_norm*np.exp(-(q/4)*np.power(np.log(freq/peak_freq),2))

def logduffcurve(freq_log,S_norm_log,q,peak_freq_log): # Duffy & Blundell 2012 curved model but fitting in log space.
	return S_norm_log - (q/4)*(freq_log - peak_freq_log)**2

# Following models add a break to the spectrum

def powlawbreak(freq, S_norm, alpha, breakfreq): # Continuous injection model.
	alpha = np.where(freq <= breakfreq, alpha, alpha+0.5)
	freq = freq / breakfreq
	return S_norm*(freq**-alpha)

def singhomobremssbreak(freq,S_norm,alpha,freq_peak,breakfreq): # Continuous injection model.
	alpha = np.where(freq <= breakfreq, alpha, alpha+0.5)
	dummyfreq = freq / breakfreq
	return S_norm*(dummyfreq**(-alpha))*np.exp(-(freq/freq_peak)**(-2.1))

def singinhomobremssbreak(freq,S_norm,alpha,p,freq_peak,breakfreq): # Continuous injection model.
	dummyalpha = np.where(freq <= breakfreq, alpha, alpha+0.5)
	dummyfreq = freq / breakfreq
	return S_norm*(p+1)*np.power((freq/freq_peak),(2.1*(p+1)))*np.power(dummyfreq,-dummyalpha)*np.power(freq_peak,alpha)*special.gammainc((p+1),(freq/freq_peak)**(-2.1))*special.gamma(p+1)

def powlawexp(freq, S_norm, alpha, breakfreq): # Exponential break
	return S_norm*np.power(freq,-alpha)*np.exp(-freq/breakfreq)

def singinhomobremssbreakexp(freq,S_norm,alpha,p,freq_peak,breakfreq): # Single inhomogeneous free-free emission model with exponential break
	return S_norm*(p+1)*(np.power((freq/freq_peak),(2.1*(p+1)))*np.power(freq,-alpha)*np.power(freq_peak,alpha)*np.exp(-freq/breakfreq)*special.gammainc((p+1),(freq/freq_peak)**(-2.1)))*special.gamma(p+1)

def singhomobremssbreakexp(freq,S_norm,alpha,freq_peak,breakfreq): # nonthermal power law spectrum absorbed by a homogeneous free-free absorbing screen with exponential break.
	return S_norm*(freq**(-alpha))*np.exp(-freq/breakfreq)*np.exp(-(freq/freq_peak)**(-2.1))

def doubhomobremssbreakexp(freq,S_norm1,S_norm2,alpha1,alpha2,freq_peak1,freq_peak2,breakfreq): # Two seperate nonthermal power-law components, each with its own homogeneous free-free absorbing screen, with exponential break.
 	return S_norm1*(freq**(-alpha1))*np.exp(-(freq/freq_peak1)**(-2.1)) + S_norm2*(freq**(-alpha2))*np.exp(-freq/breakfreq)*np.exp(-(freq/freq_peak2)**(-2.1))

def singSSAbreakexp(freq,S_norm,beta,peak_freq,breakfreq): # with exponential break
	return S_norm*((freq/peak_freq)**(-(beta-1)/2))*np.exp(-freq/breakfreq)*(1-np.exp(-(freq/peak_freq)**(-(beta+4)/2)))/((freq/peak_freq)**(-(beta+4)/2))

def doubSSAbreakexp(freq,S_norm1,S_norm2,beta1,beta2,peak_freq1,peak_freq2,breakfreq): # Only have break on high freq component. 
	return S_norm1*((freq/peak_freq1)**(-(beta1-1)/2))*(1-np.exp(-(freq/peak_freq1)**(-(beta1+4)/2)))/((freq/peak_freq1)**(-(beta1+4)/2)) + S_norm2*((freq/peak_freq2)**(-(beta2-1)/2))*np.exp(-freq/breakfreq)*(1-np.exp(-(freq/peak_freq2)**(-(beta2+4)/2)))/((freq/peak_freq2)**(-(beta2+4)/2))


