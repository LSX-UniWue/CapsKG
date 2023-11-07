
import time
import pickle
import pandas as pd
import utils
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset

# Args -- Experiment
from preparation import *
import import_classification as import_modules


print('Load data...')
# reuse data if already loaded
if args.backbone == 'w2v' or args.backbone == 'w2v_as':
    data, taskcla, vocab_size, embeddings = import_modules.dataloader.get(logger=logger, args=args)
else:
    data_filename = f"{args.task}_{args.ntasks}data.pickle"
    taskcla_filename = f"{args.task}_{args.ntasks}taskcla.pickle"
    if args.reuse_data and os.path.exists(data_filename):
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)
        with open(taskcla_filename, 'rb') as f:
            taskcla = pickle.load(f)
        # get the order from file (first time made in dataloader)
        f_name = data['task_seq_inf']['seq_file_name']
        if str(args.ntasks) not in f_name:
            f_name = f_name + f'_{args.ntasks}'
        with open(f_name, 'r') as f_random_seq:
            random_sep = f_random_seq.readlines()[args.idrandom].split()[:args.ntasks]
        ordered_data = {'ncla': data['ncla']}
        ordered_taskcla = []
        for i, task_name in enumerate(random_sep):
            task_id = data['task_seq_inf'][task_name]
            ordered_data[i] = data[task_id]
            ordered_taskcla.append((i, (taskcla[task_id][1])))  # assuming taskcla is ordered
        data = ordered_data
        taskcla = ordered_taskcla

    else:
        data, taskcla = import_modules.dataloader.get(logger=logger, args=args)
        if args.reuse_data:
            with open(data_filename, 'wb') as f:
                pickle.dump(data, f)
            with open(taskcla_filename, 'wb') as f:
                pickle.dump(taskcla, f)

print('\nTask info =', taskcla)
#
# Inits
print('Inits...')

# ----------------------------------------------------------------------
# Apply approach and network.
# ----------------------------------------------------------------------

if 'owm' in args.baseline:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'w2v' in args.baseline:
        net = import_modules.network.Net(taskcla, embeddings, args=args)
    elif args.load_trained_capsule_model:
        if torch.cuda.is_available():
            net = torch.load(args.trained_model_path)
        else:
            net = torch.load(args.trained_model_path, map_location=torch.device('cpu'))
    else:
        net = import_modules.network.Net(taskcla, args=args)

elif 'ucl' in args.baseline:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'w2v' in args.baseline:
        net = import_modules.network.Net(taskcla, embeddings, args.ratio, args=args)
        net_old = import_modules.network.Net(taskcla, embeddings, args.ratio, args=args)
    elif args.load_trained_capsule_model:
        if torch.cuda.is_available():
            net = torch.load(args.trained_model_path)
        else:
            net = torch.load(args.trained_model_path, map_location=torch.device('cpu'))
    else:
        net = import_modules.network.Net(taskcla, args=args)
        net_old = import_modules.network.Net(taskcla, args=args)

else:  # TODO my is used for my experiments
    if 'w2v' in args.baseline:
        net = import_modules.network.Net(taskcla, embeddings, args=args)
    elif args.load_trained_capsule_model:
        if torch.cuda.is_available():
            net = torch.load(args.trained_model_path)
        else:
            net = torch.load(args.trained_model_path, map_location=torch.device('cpu'))
    else:
        net = import_modules.network.Net(taskcla, args=args)

if 'net' in locals(): net = net.to(device)
if 'net_old' in locals(): net_old = net_old.to(device)
appr = import_modules.approach.Appr(net, logger=logger, taskcla=taskcla, args=args)

if not args.eval_each_step:
    resume_checkpoint(appr, net)

if args.multi_gpu and args.distributed:

    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
        world_size=n_gpu)
    net = net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank], output_device=args.local_rank)
    # TODO: distributed may be hang and stuck here

elif args.multi_gpu:
    logger.info('multi_gpu')
    net = torch.nn.DataParallel(net)
    net = net.to(device)

if args.print_report:
    utils.print_model_report(net)

# ----------------------------------------------------------------------
# Start Training.
# ----------------------------------------------------------------------
test_table = pd.DataFrame(columns=list(range(args.ntasks)))
for t, ncla in taskcla:

    if args.eval_each_step:
        args.resume_from_aux_file = base_resume_from_aux_file + 'steps' + str(t)
        args.resume_from_file = base_resume_from_file + 'steps' + str(t)
        resume_checkpoint(appr)

    logger.info('*' * 100)
    logger.info('Task {:2d} ({:s})'.format(t, data[t]['name']))
    logger.info('*' * 100)

    # if t>1: exit()

    if 'mtl' in args.baseline:
        # Get data. We do not put it to GPU
        if 'train_nsp' in data[t].keys():
            train_nsp = data[t]['train_nsp']
        else:
            train_nsp = None
        if t == 0:
            train = data[t]['train']
            valid = data[t]['valid']
            num_train_steps = data[t]['num_train_steps']

        else:
            train = ConcatDataset([train, data[t]['train']])
            valid = ConcatDataset([valid, data[t]['valid']])
            num_train_steps += data[t]['num_train_steps']
        task = t

        if t < len(taskcla) - 1: continue  # only want the last one

    else:
        # Get data
        train = data[t]['train']
        if 'train_nsp' in data[t].keys():
            train_nsp = data[t]['train_nsp']
        else:
            train_nsp = None
        valid = data[t]['valid']
        num_train_steps = data[t]['num_train_steps']
        task = t

    if args.task == 'asc':  # special setting
        if 'XuSemEval' in data[t]['name']:
            args.num_train_epochs = args.xusemeval_num_train_epochs  # 10
        else:
            args.num_train_epochs = args.bingdomains_num_train_epochs  # 30
            num_train_steps *= args.bingdomains_num_train_epochs_multiplier  # every task got refresh, *3

    if args.multi_gpu and args.distributed:
        valid_sampler = DistributedSampler(valid)  # TODO: DistributedSequentialSampler
        valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size)
    else:
        valid_sampler = SequentialSampler(valid)
        valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size, pin_memory=True)

    if args.resume_model and t < args.resume_from_task: continue  # resume. dont forget to copy the forward results

    if args.multi_gpu and args.distributed:
        train_sampler = DistributedSampler(train)
        if train_nsp:
            train_nsp_sampler = DistributedSampler(train_nsp)
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size)
        if train_nsp:
            train_nsp_dataloader = DataLoader(train_nsp, sampler=train_nsp_sampler, batch_size=args.train_batch_size)
    else:
        train_sampler = RandomSampler(train)
        if train_nsp:
            train_nsp_sampler = RandomSampler(train_nsp)
        train_dataloader = DataLoader(train, sampler=train_sampler, batch_size=args.train_batch_size, pin_memory=True#,num_workers=4
                                      )
        if train_nsp:
            train_nsp_dataloader = DataLoader(train_nsp, sampler=train_nsp_sampler, batch_size=args.train_batch_size,
                                              pin_memory=True)

    logger.info('Start Training and Set the clock')
    tstart = time.time()

    if not args.eval_only:
        if args.task in extraction_tasks:
            label_list = data[t]['label_list']
            appr.train(task, train_dataloader, valid_dataloader, num_train_steps, train, valid, label_list)
        else:
            # Train
            print('train')
            if train_nsp:
                appr.train(task, train_dataloader, valid_dataloader, train_nsp_dataloader, num_train_steps, train,
                           valid, train_nsp)
            else:
                if args.multi_gpu and args.distributed:
                    test = data[t]['test']
                    test_sampler = DistributedSampler(test)
                    test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
                else:
                    test = data[t]['test']
                    test_sampler = SequentialSampler(test)
                    test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
                appr.train(task, train_dataloader, valid_dataloader, test_dataloader, num_train_steps, train, valid)

    print('-' * 100)

    if args.exit_after_first_task:  # sometimes we want to fast debug or estimate the excution time
        # TODO: consider save to file and print them out
        logger.info('[Elapsed time per epochs = {:.1f} s]'.format((time.time() - tstart)))
        logger.info('[Elapsed time per epochs = {:.1f} min]'.format((time.time() - tstart) / (60)))
        logger.info('[Elapsed time per epochs = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
        if 'kim' not in args.baseline and 'mlp' not in args.baseline and 'cnn' not in args.baseline:
            if 'asc' in args.task:
                pre_define_num_epochs = 30  # non-semeval estimation
            elif 'dsc' in args.task:
                pre_define_num_epochs = 20
            elif 'newsgroup' in args.task:
                pre_define_num_epochs = 10
            logger.info('[Elapsed time per tasks = {:.1f} s]'.format((time.time() - tstart) * pre_define_num_epochs))
            logger.info(
                '[Elapsed time per tasks = {:.1f} min]'.format(((time.time() - tstart) / (60)) * pre_define_num_epochs))
            logger.info('[Elapsed time per tasks = {:.1f} h]'.format(
                ((time.time() - tstart) / (60 * 60)) * pre_define_num_epochs))
        else:
            if 'asc' in args.task:
                additional = 3  # different size for asc tasks
            else:
                additional = 1
            logger.info('[Elapsed time per tasks = {:.1f} s]'.format(
                (time.time() - tstart) * 50 * additional))  # estimation for early stopping
            logger.info(
                '[Elapsed time per tasks = {:.1f} min]'.format(((time.time() - tstart) / (60)) * 50 * additional))
            logger.info(
                '[Elapsed time per tasks = {:.1f} h]'.format(((time.time() - tstart) / (60 * 60)) * 50 * additional))
        exit()

    if args.save_each_step:
        args.model_path = base_model_path + 'steps' + str(t)
        base_aux_model_path = base_model_path + '_aux_model_'
        args.aux_model_path = base_aux_model_path + 'steps' + str(t)

    if args.save_model:
        print('save model')
        torch.save({'model_state_dict': appr.model.state_dict()}, args.model_path)

        # for GEM
        if hasattr(appr, 'buffer'):
            torch.save(appr.buffer, args.model_path + '_buffer')  # not in state_dict
        if hasattr(appr, 'grad_dims'):
            torch.save(appr.grad_dims, args.model_path + '_grad_dims')  # not in state_dict
        if hasattr(appr, 'grads_cs'):
            torch.save(appr.grads_cs, args.model_path + '_grads_cs')  # not in state_dict
        if hasattr(appr, 'grads_da'):
            torch.save(appr.grads_da, args.model_path + '_grads_da')  # not in state_dict
        if hasattr(appr, 'history_mask_pre'):
            torch.save(appr.history_mask_pre, args.model_path + '_history_mask_pre')  # not in state_dict
        if hasattr(appr, 'similarities'):
            torch.save(appr.similarities, args.model_path + '_similarities')  # not in state_dict
        if hasattr(appr, 'check_federated'):
            torch.save(appr.check_federated, args.model_path + '_check_federated')  # not in state_dict

        if 'aux_net' in args and args.aux_net:
            torch.save({'model_state_dict': appr.aux_model.state_dict()}, args.aux_model_path)
            if hasattr(appr, 'mask_pre'):
                torch.save(appr.mask_pre, args.aux_model_path + '_mask_pre')  # not in state_dict
            if hasattr(appr, 'mask_back'):
                torch.save(appr.mask_back, args.aux_model_path + '_mask_back')
        else:
            if hasattr(appr, 'mask_pre'):
                torch.save(appr.mask_pre, args.model_path + '_mask_pre')  # not in state_dict
            if hasattr(appr, 'mask_back'):
                torch.save(appr.mask_back, args.model_path + '_mask_back')

    # ----------------------------------------------------------------------
    # Start Testing.
    # ----------------------------------------------------------------------
    print('Testing')
    if args.eval_forward or (t == (args.ntasks - 1)):
        if args.unseen and args.eval_each_step:  # we want to test every one for unseen
            test_set = args.ntasks
        else:
            test_set = t + 1

        row = np.zeros(args.ntasks)
        for u in range(test_set):

            # test only the diagonal and last row (results when task is first trained and in final model)
            if u != t and t != (args.ntasks - 1):
                row[u] = 0.0
                acc[t, u] = 0.0
                lss[t, u] = 0.0
                f1_macro[t, u] = 0.0
                continue

            test = data[u]['test']

            if args.multi_gpu and args.distributed:
                test_sampler = DistributedSampler(test)
                test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)
            else:
                test_sampler = SequentialSampler(test)
                test_dataloader = DataLoader(test, sampler=test_sampler, batch_size=args.eval_batch_size)

            if args.task in classification_tasks:  # classification task

                if 'kan' in args.baseline:
                    test_loss, test_acc, test_f1_macro = appr.eval(u, test_dataloader, test, which_type='mcl',
                                                                   trained_task=t)
                    logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'],
                                                                                                        test_loss,
                                                                                                        100 * test_acc))
                elif 'cat' in args.baseline:
                    valid = data[u]['valid']
                    valid_sampler = SequentialSampler(valid)
                    valid_dataloader = DataLoader(valid, sampler=valid_sampler, batch_size=args.eval_batch_size,
                                                  pin_memory=True)
                    test_loss, test_acc, test_f1_macro = appr.eval(u, test_dataloader, valid_dataloader, trained_task=t,
                                                                   phase='mcl')
                    logger.info('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'],
                                                                                                        test_loss,
                                                                                                        100 * test_acc))

                else:
                    if args.task in ['wn18', 'yago','fbl']:
                        test_loss, test_acc, test_f1_macro = appr.eval(u, test_dataloader, test, trained_task=t,
                                                                       filter_eval=True)
                    elif args.log_pred:
                        test_loss, test_acc, test_f1_macro = appr.eval(u, test_dataloader, test, trained_task=t,
                                                                       log_pred=args.log_pred)
                    else:
                        test_loss, test_acc, test_f1_macro = appr.eval(u, test_dataloader, test, trained_task=t)
                    row[u] = test_acc

                acc[t, u] = test_acc
                lss[t, u] = test_loss
                f1_macro[t, u] = test_f1_macro

        # Done
        print('*' * 100)
        print('Accuracies =')
        for i in range(acc.shape[0]):
            print('\t', end='')
            for j in range(acc.shape[1]):
                print('{:5.1f}% '.format(100 * acc[i, j]), end='')
            print()
        print('*' * 100)
        print('Done!')

        print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
