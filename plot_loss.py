"""
Plot GAN training progress.
Author: Sara Mathieson
Date: 6/24/20
"""
​
# python imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import optparse
import sys
from matplotlib.pyplot import figure
​
from decimal import Decimal
​
# our imports
import util
​
def parse_args():
   """Parse command line arguments."""
   parser = optparse.OptionParser(description='plot PG-GAN result')
​
   parser.add_option('-f', '--input', type='string', \
       help='path to input file of output results')
   parser.add_option('-o', '--output', type='string', \
       help='path to output figure (optional)')
​
   (opts, args) = parser.parse_args()
​
   mandatories = ['input']
   for m in mandatories:
       if not opts.__dict__[m]:
           print('mandatory option ' + m + ' is missing\n')
           parser.print_help()
           sys.exit()
​
   return opts
​
def parse_mini_lst(mini_lst):
   return [float(x.replace("[",'').replace("]",'').replace(",",'')) for x in \
       mini_lst]
​
def add_to_lst(total_lst, mini_lst):
   assert len(total_lst) == len(mini_lst)
   for i in range(len(total_lst)):
       total_lst[i].append(mini_lst[i])
​
def analyze_acc_lsts(real_acc_lst, fake_acc_lst):
   idx_lst = []
   for i in range(len(real_acc_lst)):
       b1 = 0.4 <= real_acc_lst[i] <= 0.6
       b2 = 0.4 <= fake_acc_lst[i] <= 0.6
       if b1 and b2:
           idx_lst.append(i)
           if i-1 in idx_lst and i-2 in idx_lst:
               return i
​
   return None
​
def parse_output(filename):
   f = open(filename,'r')
​
   # list of lists, one for each param
   read_param = False
   name_lst = []
   param_lst_all = []
   true_lst = []
   min_lst = []
   max_lst = []
​
   gen_loss_lst = []
   disc_loss_lst = []
​
   real_acc_lst = []
   fake_acc_lst = []
   #num_param = None
   #param_str = None
​
   gen_loss = None
   disc_loss = None
   real_acc = None
   fake_acc = None
​
   for line in f:
​
       if line.startswith("ITER"):
           tokens = line.split()
           iter = int(tokens[1])
           #if iter > 0:
           gen_loss_lst.append(gen_loss)
           disc_loss_lst.append(disc_loss)
           real_acc_lst.append(real_acc)
           fake_acc_lst.append(fake_acc)
​
       elif line.startswith("NAME"):
           read_param = True
​
       elif read_param:
           param = line.split()
           name = param[0]
           print(name)
           name_lst.append(name)
           true_lst.append(float(param[1]))
           min_lst.append(float(param[2]))
           max_lst.append(float(param[3]))
           param_lst_all.append([])
           read_param = False
​
       elif "Epoch 100" in line:
           tokens = line.split()
           disc_loss = float(tokens[3][:-1])
           real_acc = float(tokens[6][:-1])/100
           fake_acc = float(tokens[9])/100
​
       '''if line.startswith("{"):
           tokens = line.split()
           param_str = tokens[5][1:-2]
​
       if "params, test_acc" in line:
           tokens = line.split()
​
           # set up total param lists
           num_param = len(tokens)-3
           for i in range(num_param):
               param_lst_all.append([])
​
           # parse current params and add to each list
           mini_lst = parse_mini_lst(tokens[2:2+num_param])
           add_to_lst(param_lst_all, mini_lst)
​
           # record test accuracy
           test_acc_lst.append(float(tokens[-1]))'''
​
       #if "i, T" in line: # TODO toggle
       if "T, p_accept" in line:
           tokens = line.split()
​
           # parse current params and add to each list
           num_param = len(true_lst)
           mini_lst = parse_mini_lst(tokens[-1-num_param:-1]) # toggle -1 or -5
           add_to_lst(param_lst_all, mini_lst)
​
           # record test accuracy
           gen_loss = float(tokens[-1]) # toggle -1 or -5
​
   f.close()
​
   if gen_loss_lst[0] == None:
       print("None!")
       gen_loss_lst[0] = gen_loss_lst[1]
   print(len(gen_loss_lst), len(param_lst_all[0]))
   if len(gen_loss_lst) == len(param_lst_all[0]) + 1:
       gen_loss_lst = gen_loss_lst[1:]
       disc_loss_lst = disc_loss_lst[1:]
       real_acc_lst = real_acc_lst[1:]
       fake_acc_lst = fake_acc_lst[1:]
   assert len(gen_loss_lst) == len(param_lst_all[0])
   return name_lst, gen_loss_lst, disc_loss_lst, real_acc_lst, fake_acc_lst, true_lst, min_lst, max_lst, param_lst_all # param_str, param_lst_all, test_acc_lst
​
def plot_param(subplot, param_lst, test_acc_lst, name, true):
   plt.subplot(subplot)
   cm = plt.get_cmap('rainbow')
   num_iter = len(param_lst)-1
​
   # line showing 0.5 goal
   plt.plot([min(param_lst), max(param_lst)], [0.5,0.5], 'k--', lw=0.5, \
       label=None)
​
   # set up heatmap
   color = [(num_iter-i) for i in range(num_iter+1)]
   heatmap = plt.scatter(param_lst, test_acc_lst, s=10, c=color, cmap=cm, \
       vmin=0, vmax=num_iter, label="GAN iterations")
​
   # heatmap legend
   #cb = plt.colorbar(heatmap)
   #cb.set_ticks([0,num_iter])
   #cb.set_ticklabels(["end", "begin"])
​
   # inferred and true
   # try a different way
   below_half = [x <= 0.5 for x in test_acc_lst]
   infer = param_lst[-1]
   '''for i in range(len(param_lst)):
       if below_half[i]:
           infer = param_lst[i] # latest below 0.5'''
   #infer = param_lst[-1]
   #infer = param_lst[min_test_acc_idx]
   if 0.01 < infer < 1:
       infer_str = '%.3f' % Decimal(str(infer))
       true_str = str(true)
   elif infer < 0.01:
       infer_str = '%.2e' % Decimal(str(infer))
       true_str = str(true)
   else:
       infer_str = str(round(infer))
       true_str = str(round(true))
   plt.plot([infer, infer], [min(test_acc_lst), max(test_acc_lst)], \
       label="inferred: " + infer_str)
   plt.plot([true, true], [min(test_acc_lst), max(test_acc_lst)], \
       label="true: " + true_str)
​
   # setup plot
   plt.xlabel("parameter: " + name)
   #plt.ylabel("test accuracy")
   plt.title("pg-gan training process")
   plt.legend()
​
​
def main():
   opts = parse_args()
   name_lst, gen_loss_lst, disc_loss_lst, real_acc_lst, fake_acc_lst, true_lst, min_lst, max_lst, param_lst_all = parse_output(opts.input)
​
   # set up main variables
   num_iter = len(gen_loss_lst)
   print("num iter", num_iter)
   stop = analyze_acc_lsts(real_acc_lst, fake_acc_lst)
   print("stop idx", stop)
   num_param = len(true_lst)
​
   plt.figure(num=None, figsize=(14, 10)) #figsize=(14, 4))
   #num_param = 0
​
​
   # plot losses
   subplot = int(str(num_param+2) + '11')
   #subplot = int(str(4) + '11')
   plt.subplot(subplot)
   plt.xticks([])
   plt.ylabel("loss",fontsize=14)
​
   # multiply gen loss by 2 since half as many examples
   plt.plot(range(num_iter), [x*2 for x in gen_loss_lst], 'g')
   plt.plot(range(num_iter), disc_loss_lst, 'm')
   ax = plt.gca()
   ax.set_facecolor('whitesmoke')
   plt.legend(["generator loss", "discriminator loss"], loc=10)
​
   # plot accuracies
   subplot = int(str(num_param+2) + '12')
   #subplot = int(str(4) + '12')
   plt.subplot(subplot)
   plt.ylabel("accuracy",fontsize=14)
   plt.xticks([])
   #plt.xlabel("training iteration")
​
   plt.plot(range(num_iter), fake_acc_lst)
   plt.plot(range(num_iter), real_acc_lst)
   plt.plot(range(num_iter), [0.5]*num_iter, 'k--', lw=0.5)
   ax = plt.gca()
   ax.set_facecolor('whitesmoke')
   #plt.plot([stop, stop], [0, 1], 'k--', lw=0.5)
   plt.legend(["generated accuracy", "training accuracy"], loc=10)
​
   #final = (fake_acc_lst[-1] + real_acc_lst[-1])/2
   #print("final avg acc", final, fake_acc_lst[-1], real_acc_lst[-1])
​
​
   for i in range(num_param):
       subplot = int(str(num_param+2) + '1' + str(i+3))
       plt.subplot(subplot)
       name = name_lst[i]
       plt.ylabel(name,fontsize=14)
       param_values = param_lst_all[i]
​
​
       #true = '{:0.3e}'.format(true_lst[i])
       true = true_lst[i]
       infer = '{:0.2e}'.format(param_values[-1])
       if name == "mig":
           infer = round(param_values[-1],3)
       elif name not in ["mig", "reco","mut"]:
           true = round(true_lst[i])
           infer = round(param_values[-1])
​
​
       plt.plot(range(num_iter), param_values, label="inferred: " + str(infer))
       plt.plot(range(num_iter), [min_lst[i]]*num_iter, 'k--', lw=0.5)
       plt.plot(range(num_iter), [true_lst[i]]*num_iter, label="true: " + str(true))
       plt.plot(range(num_iter), [max_lst[i]]*num_iter, 'k--', lw=0.5)
​
       if i != num_param-1:
           plt.xticks([])
       else:
           plt.xlabel("training iteration",fontsize=14)
​
       ax = plt.gca()
       ax.set_facecolor('whitesmoke')
​
       plt.legend(loc=10)
       #plt.plot([stop, stop], [min_lst[i], max_lst[i]], 'k--', lw=0.5)
       #plot_param(subplot, param_lst_all[i], test_acc_lst, parameters[i].name,\
       #    parameters[i].value)
​
​
   gs1 = gridspec.GridSpec(14, 8)
   plt.tight_layout()
   gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes.
   if opts.output != None:
       plt.savefig(opts.output, format="pdf", dpi=600)
   else:
       plt.show()
​
main()