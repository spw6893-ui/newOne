import os
import yaml
import argparse
import torch
from datetime import datetime
from pathlib import Path

from fqf_iqn_qrdqn.agent import QRQCMAgent, IQCMAgent, FQCMAgent
from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType
from alphagen_qlib.crypto_data import CryptoData
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.models.alpha_pool import AlphaPool
from alphagen.rl.env.wrapper import AlphaEnv


def run(args):
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / 'qcm_config' / f'{args.model}.yaml'

    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # Create environments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    close = Feature(FeatureType.CLOSE)

    # Target: predict future return (e.g., 20 periods ahead)
    target = Ref(close, -args.target_periods) / close - 1

    # Symbol groups: 'top10', 'top20', 'all'
    symbols = args.symbols

    # Load crypto data
    data_dir = str((base_dir / 'AlphaQCM_data' / 'crypto_data').resolve())
    data_train = CryptoData(
        symbols=symbols,
        start_time=args.train_start,
        end_time=args.train_end,
        timeframe=args.timeframe,
        data_dir=data_dir
    )
    data_valid = CryptoData(
        symbols=symbols,
        start_time=args.valid_start,
        end_time=args.valid_end,
        timeframe=args.timeframe,
        data_dir=data_dir
    )
    data_test = CryptoData(
        symbols=symbols,
        start_time=args.test_start,
        end_time=args.test_end,
        timeframe=args.timeframe,
        data_dir=data_dir
    )

    train_calculator = QLibStockDataCalculator(data_train, target)
    valid_calculator = QLibStockDataCalculator(data_valid, target)
    test_calculator = QLibStockDataCalculator(data_test, target)

    train_pool = AlphaPool(
        capacity=args.pool,
        calculator=train_calculator,
        ic_lower_bound=None,
        l1_alpha=5e-3
    )
    train_env = AlphaEnv(pool=train_pool, device=device, print_expr=True)

    # Specify the directory to log
    name = args.model
    time = datetime.now().strftime("%Y%m%d-%H%M")

    if name in ['qrdqn', 'iqn']:
        log_dir = os.path.join(
            str((base_dir / 'AlphaQCM_data' / 'crypto_logs').resolve()),
            f'{symbols}_{args.timeframe}',
            f'pool_{args.pool}_QCM_{args.std_lam}',
            f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}"
        )
    elif name == 'fqf':
        log_dir = os.path.join(
            str((base_dir / 'AlphaQCM_data' / 'crypto_logs').resolve()),
            f'{symbols}_{args.timeframe}',
            f'pool_{args.pool}_QCM_{args.std_lam}',
            f"{name}-seed{args.seed}-{time}-N{config['N']}-lr{config['quantile_lr']}-per{config['use_per']}-gamma{config['gamma']}-step{config['multi_step']}"
        )

    # Create the agent and run
    if name == 'qrdqn':
        agent = QRQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )
    elif name == 'iqn':
        agent = IQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )
    elif name == 'fqf':
        agent = FQCMAgent(
            env=train_env,
            valid_calculator=valid_calculator,
            test_calculator=test_calculator,
            log_dir=log_dir,
            seed=args.seed,
            std_lam=args.std_lam,
            cuda=True,
            **config
        )

    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qrdqn',
                        choices=['qrdqn', 'iqn', 'fqf'],
                        help='Model type')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--pool', type=int, default=20,
                        help='Alpha pool capacity')
    parser.add_argument('--std-lam', type=float, default=1.0,
                        help='Standard deviation lambda')
    parser.add_argument('--symbols', type=str, default='top10',
                        choices=['top10', 'top20', 'top100', 'all'],
                        help='Symbol group to trade')
    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                        help='Candle timeframe')
    parser.add_argument('--target-periods', type=int, default=20,
                        help='Number of periods ahead to predict')
    parser.add_argument('--train-start', type=str, default='2020-01-01',
                        help='Train start (inclusive)')
    parser.add_argument('--train-end', type=str, default='2023-12-31',
                        help='Train end (inclusive)')
    parser.add_argument('--valid-start', type=str, default='2024-01-01',
                        help='Validation start (inclusive)')
    parser.add_argument('--valid-end', type=str, default='2024-06-30',
                        help='Validation end (inclusive)')
    parser.add_argument('--test-start', type=str, default='2024-07-01',
                        help='Test start (inclusive)')
    parser.add_argument('--test-end', type=str, default='2024-12-31',
                        help='Test end (inclusive)')
    args = parser.parse_args()
    run(args)
