import torch
import numpy as np
import glob
from lib.train import Train
from lib.utils import smooth
from sklearn.externals import joblib

def plot_train_val(patterns, fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    dummy_trainer = Train(None, None, None, None)
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['train_losses']
            dummy_trainer.all_losses = l
            plt.plot(dummy_trainer.smooth_loss(), label=name)
            
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['val_losses']
            dummy_trainer.val_losses = l
            plt.plot(dummy_trainer.smooth_valloss(), label=name)

    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
def plot_fill(lines, x=None, color='b', label='default'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for l in lines:
        if x is not None:
            plt.plot(x, l, color=color, alpha=0.2)
        else:
            plt.plot(l, color=color, alpha=0.2)
    
    # lines may not have the same length
    max_length = max([len(l) for l in lines])
    middle_line = np.zeros(max_length)    
    for i in range(max_length):
        middle_line[i] = np.percentile([l[i] for l in lines if len(l) > i], 50)
        
    if x is not None:
        plt.plot(x, middle_line, color=color, label=label)
    else:
        plt.plot(middle_line, color=color, label=label)
    
def get_train_val_curves(pattern):
    dummy_trainer = Train(None, None, None, None)
    tr_curves = []
    val_curves = []
    name = ""
    for cpt_fn in glob.glob(pattern):
        cpt = torch.load(cpt_fn)
        name = cpt_fn.split('.')[0].split('/')[-1]
        dummy_trainer.all_losses = cpt['train_losses']
        dummy_trainer.val_losses = cpt['val_losses']
        
        tr_curves.append(dummy_trainer.smooth_loss())
        val_curves.append(dummy_trainer.smooth_valloss())
    return tr_curves, val_curves, name

def plot_train_val_multiple(patterns, colors=['blue', 'orange', 'green', 'red', 
                                              'purple', 'brown', 'pink', 'gray'], 
                            fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(tr_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(val_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation loss (RMSE)', fontsize=fontsize)
    plt.grid()
    plt.show()

        
def plot_best_test(train_pattern, smooth_window=1, title=None, ylim=None, xlim=None, 
                   methods=None, method2label=None, lr=None, start=0, end=-1, 
                   default_setting=None, ylabel=None, q_low=25, q_high=75):
    import matplotlib.pyplot as plt        
    assert 'train_losses' in train_pattern, 'train_losses must appear in train_pattern'
    methods_settings = {}
    val_settings = {}
    test_settings = {}
    accs = {} # accs are based on best validation performance
    best_val_test_error = {}
    for fn in sorted(glob.glob(train_pattern)):
        method_setting = fn.split('/')[-1].split('^')[0].split('-')
        method, setting = method_setting[0], '-'.join(method_setting[1:])
        
        method_lr = '-'.join(method_setting[1:3]) \
                    if method_setting[1] == '1e' else method_setting[1]
        if lr is not None and lr != float(method_lr):
            continue
        if methods is not None and method not in methods:
            continue
        
        if method not in methods_settings:
            methods_settings[method] = {}
            val_settings[method] = {}
            test_settings[method] = {}
            accs[method] = {}
        if setting not in methods_settings[method]:
            methods_settings[method][setting] = []
            val_settings[method][setting] = []
            test_settings[method][setting] = []
            accs[method][setting] = []
        
        tr_loss = smooth(joblib.load(fn), smooth_window)
        val_error = smooth(joblib.load(fn.replace('train_losses', 'val_errors')),
                           smooth_window)
        test_error = smooth(joblib.load(fn.replace('train_losses', 'test_errors')),
                            smooth_window)
        #if len(tr_loss)==10: print(fn)
            
        methods_settings[method][setting].append(tr_loss)
        val_settings[method][setting].append(val_error)
        test_settings[method][setting].append(test_error)

        # change the following two to see different evaluation
        best_error_index = np.argmin(val_error)
        # best_error_index = -1
        min_val_error = val_error[best_error_index]
        
        accs[method][setting].append(test_error[best_error_index])

        if best_val_test_error.get(method) is None:
            best_val_test_error[method] = (min_val_error,
                                           test_error[best_error_index], fn)
        else:
            val_best_error, test_best_error, fn_best = best_val_test_error[method]
            if val_best_error >= min_val_error:
                val_best_error = min_val_error
                test_best_error = test_error[best_error_index]
                best_val_test_error[method] = (val_best_error, test_best_error, fn)
            
    for method, setting_dict in val_settings.items():

        val_best_error, test_best_error, fn_best = best_val_test_error[method]   
        print('{} test: {:.3f} val: {:.3f} fn:{}'.format(method, test_best_error,
                                                     val_best_error, fn_best))
        
        setting_areas = []
        for setting, v in setting_dict.items():
            # find the smallest area
            max_len = max([len(a) for a in v])
            v = [a for a in v if len(a) == max_len]
            area = np.mean([a[start:end] for a in v])
            setting_areas.append((setting, area))
        setting = sorted(setting_areas, key=lambda x: x[1])[0][0]
            
        def plot_setting():
            v = test_settings[method][setting]
            max_len = max([len(a) for a in v])
            v = [a for a in v if len(a) == max_len]

            # print('{}: {:.3f} ({:.2f}) {} runs'.format(method,
            #                                            np.mean(accs[method][setting]), 
            #                                            np.std(accs[method][setting]),
            #                                            len(v)))
            
            if method2label is None:
                label = method + '-' + setting
            else:
                label = method2label.get(method, method)
            #p = plt.plot(sum(v) / len(v), label=method + '-' + setting)#, c=colors[i]
            p = plt.plot(np.percentile(v, 50, 0), label=label, ls='--')
            plt.fill_between(np.arange(len(np.percentile(v, q_low, 0))),
                             np.percentile(v, q_low, 0), np.percentile(v, q_high, 0), 
                             alpha=0.1, color=p[-1].get_color())
            
            if default_setting and method in default_setting and\
            default_setting[method] in test_settings[method]:
                v = test_settings[method][default_setting[method]]
                max_len = max([len(a) for a in v])
                v = [a for a in v if len(a) == max_len]            
                
                p = plt.plot(np.percentile(v, 50, 0), ls='-', c=p[-1].get_color())
                plt.fill_between(np.arange(len(np.percentile(v, q_low, 0))),
                                 np.percentile(v, q_low, 0),
                                 np.percentile(v, q_high, 0), 
                                 alpha=0.1, color=p[-1].get_color())
                
        plot_setting()
            
    plt.legend()
    if title:
        plt.title(title, fontsize=15)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ylabel = ylabel or "error"
    plt.ylabel('test {}'.format(ylabel), fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.grid()
    plt.show()     

def plot_best(pattern, smooth_window=1, title=None, ylim=None, xlim=None, 
              methods=None, method2label=None, lr=None, report_result=False, start=0,
              end=-1, default_setting=None, q_low=25, q_high=75):
    import matplotlib.pyplot as plt    
    methods_settings = {}
    accs = {}
    for fn in sorted(glob.glob(pattern)):
        method_setting = fn.split('/')[-1].split('^')[0].split('-')
        method, setting = method_setting[0], '-'.join(method_setting[1:])
        
        method_lr = '-'.join(method_setting[1:3]) if method_setting[1] == '1e' else method_setting[1]
        if lr is not None and lr != float(method_lr):
            continue
        if methods is not None and method not in methods:
            continue
        
        if method not in methods_settings:
            methods_settings[method] = {}
            accs[method] = {}
        if setting not in methods_settings[method]:
            methods_settings[method][setting] = []
            accs[method][setting] = []
        
        tr_loss = smooth(joblib.load(fn), smooth_window)
            
        methods_settings[method][setting].append(tr_loss)
        accs[method][setting].append(float(fn.split('/')[-1].split('^')[1]))
        
    if len(methods_settings.keys()) > 1: # plot the best one, smallest area under the curve
        for method, setting_dict in methods_settings.items():
            setting_areas = []
            for setting, v in setting_dict.items():
                # find the smallest area
                max_len = max([len(a) for a in v])
                v = [a for a in v if len(a) == max_len]            
                #min_len = min([len(a) for a in v])
                #end_ = min_len if end == -1 else min(end, min_len)
                #area = np.mean([a[start:end_] for a in v])
                area = np.mean([a[start:end] for a in v])
                setting_areas.append((setting, area))
            setting = sorted(setting_areas, key=lambda x: x[1])[0][0]

            v = setting_dict[setting]
            max_len = max([len(a) for a in v])
            v = [a for a in v if len(a) == max_len]                        
            
            print('{}: {:.2f}% ({:.2f}) {} runs'.format(method, np.mean(accs[method][setting]), 
                                                np.std(accs[method][setting]),
                                                len(v)))
            
            if method2label is None:
                label = method + '-' + setting
            else:
                label = method2label.get(method, method)
            #p = plt.plot(sum(v) / len(v), label=method + '-' + setting)#, c=colors[i]) 
            p = plt.plot(np.percentile(v, 50, 0), label=label, ls='--')
            plt.fill_between(np.arange(len(np.percentile(v, q_low, 0))),
                             np.percentile(v, q_low, 0), np.percentile(v, q_high, 0), 
                             alpha=0.1, color=p[-1].get_color())
            
            if default_setting and method in default_setting and default_setting[method] in setting_dict:
                v = setting_dict[default_setting[method]]
                max_len = max([len(a) for a in v])
                v = [a for a in v if len(a) == max_len]                        
                
                p = plt.plot(np.percentile(v, 50, 0), ls='-', c=p[-1].get_color())
                plt.fill_between(np.arange(len(np.percentile(v, q_low, 0))),
                                 np.percentile(v, q_low, 0),
                                 np.percentile(v, q_high, 0), 
                                 alpha=0.1, color=p[-1].get_color())
                        
    else: # plot all
        for method, setting_dict in methods_settings.items():
            for setting, v in setting_dict.items():                
                max_len = max([len(a) for a in v])
                v = [a for a in v if len(a) == max_len]

                print('{}: {:.2f}% ({:.2f}) {} runs'.format(method,
                                                            np.mean(accs[method][setting]), 
                                                            np.std(accs[method][setting]), 
                                                            len(v)))

                
                
                if method2label is None:
                    label = method + '-' + setting
                else:
                    label = method2label.get(method,method) + '-' + setting
                
                p = plt.plot(np.percentile(v, 50, 0), label=label)
                plt.fill_between(np.arange(len(np.percentile(v, q_low, 0))),
                                 np.percentile(v, q_low, 0),
                                 np.percentile(v, q_high, 0), 
                                 alpha=0.1, color=p[-1].get_color())                
        
    plt.legend()
    if title:
        plt.title(title, fontsize=15)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.ylabel(pattern.split("*")[-1], fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.grid()
    plt.show()     
    
# default_setting = {
#     'SGD': '0.1', 'Diff': '0.001', 'Adam': '0.001', 
#     'CrossBound': '0.001', 'AdaBound': '0.001', 'CrossAdaBound': '0.001', 'Swats': '0.001'
# }

#'AlphaSGD(1,1)', # worse b/c alpha ratio too large, especially at earlier layers, lr=1
#'AdamC1(1,1)', # worse b/c step size 0, alpha ratio too low, thus curvature high, lr=0.001
#'AdamC2(1,1)', # worse b/c step size 0, lr=1e-4, but alpha ratio is high, thus grad too small 
#'AlphaDiff(0,1)', # worse 1e-4, not learning at all
#'AlphaAdam(1,0)', # worse 1e-3, not learning at all, extremely low gradient

# methods = ['AdamC1(1,1)', 'Adam', 'AdamC2(1,1)'] # which variance to use?
# methods = ['AlphaDiff(1,1)', 'AlphaAdam(1,1)', 'SGD'] # what to use?
# methods = ['AlphaDiff(1,1)', 'AlphaDiff(0,1)', 'AlphaDiff(1,0)'] # which is dominant
# methods = ['AlphaAdam(1,1)', 'AlphaAdam(0,1)', 'AlphaAdam(1,0)']
# methods = ['AlphaDiff(0,1)', 'AlphaDiff(1,0)', 'AlphaAdam(0,1)', 'AlphaAdam(1,0)']
# methods = ['Swats', 'AdaBound', 'CrossAdaBound', 'CrossBound'] # when to switch?
# methods = ['Swats', 'AdaBound', 'CrossBound'] # when to switch?
# methods = ['AlphaDiff(1,1)', 'AlphaAdam(1,1)', 'CrossBound'] # best comparison
# methods = ['Adam', 'Diff', 'SGD'] # what to use
# methods = ['Adam', 'Diff', 'CrossBound'] # best

                                
        


