from model import SA_MVFormer

def _convert_view_weights(view_weight):
    if view_weight is None:
        return None
    if not (0 <= view_weight <= 1):
        raise ValueError(f"View weight must be between 0 and 1, got {view_weight}")
    return [1.0 - view_weight, view_weight]

def parse_method(args, n, c, d, device):
    base_kwargs = dict(
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        in_dropout=args.in_dropout,
        dropout=args.dropout,
        global_dropout=args.global_dropout,
        heads=args.num_heads,
        beta=args.beta,
        pre_ln=args.pre_ln,
        fusion_method=args.fusion_method,
        fusion_gnn_type=args.fusion_gnn_type,
        fusion_gnn_heads=args.fusion_gnn_heads,
        fusion_graph_weight=args.fusion_graph_weight,
        fusion_dropout=args.fusion_dropout,
        fusion_feature_source=args.fusion_feature_source,
        fusion_gnn_layer=args.fusion_gnn_layer,
        fusion_gcn_residual=args.fusion_gcn_residual,
        fusion_norm_before=args.fusion_norm_before,
        # view fusion controls (original + KNN)
        view_fusion_method=getattr(args, 'view_fusion_method', 'weighted'),
        view_weights=_convert_view_weights(getattr(args, 'view_weights', None)),
        view_knn_k=getattr(args, 'view_knn_k', 10),
        view_knn_metric=getattr(args, 'view_knn_metric', 'cosine'),
        knn_no_dropout=getattr(args, 'knn_no_dropout', False),
        view_gate_init=getattr(args, 'view_gate_init', 0.0),

    )

    model = SA_MVFormer(d, args.hidden_channels, c, **base_kwargs).to(device)
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_epochs', type=int, default=1000,
                        help='warmup epochs for local attention')
    parser.add_argument('--global_epochs', type=int, default=1000,
                        help='epochs for local-to-global attention')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    # model
    parser.add_argument('--method', type=str, default='poly')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7,
                        help='number of layers for local attention')
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--beta', type=float, default=-1.0,
                        help='SA_MVFormer beta initialization')
    parser.add_argument('--pre_ln', action='store_true')

    parser.add_argument('--no_resume', action='store_true',
                        help='No checkpoint loaded; training from scratc')

    # fusion branch
    parser.add_argument('--fusion_method', type=str, default='weighted', choices=['concat', 'weighted'],
                        help='fusion method inside fusionblock')
    parser.add_argument('--fusion_gnn_type', type=str, default='gcn', choices=['gcn', 'sage'],
                        help='GNN type for fusion block')
    parser.add_argument('--fusion_gnn_heads', type=int, default=4,
                        help='number of heads if fusion_gnn_type is gat')
    parser.add_argument('--fusion_graph_weight', type=float, default=0.8,
                        help='graph branch weight when fusion_method is weighted')
    parser.add_argument('--fusion_dropout', type=float, default=None,
                        help='dropout used inside fusion block; default to dropout if None')
    parser.add_argument('--fusion_feature_source', type=str, default='proj', choices=['proj','local'],
                        help='feature source sent into GNN fusion (proj/local)')
    parser.add_argument('--fusion_norm_before', action='store_true',
                        help='apply LayerNorm to GNN and attention outputs before fusion')
    parser.add_argument('--fusion_gnn_layer', type=int, default=1,
                        help='number of layers for fusion GNN (effective for gcn/gat/sage; ignored for dense)')
    parser.add_argument('--fusion_gcn_residual', action='store_true', default=False,
                        help='use residual connections in GCN fusion block (only for gcn type)')

    # view fusion controls (original + KNN)
    parser.add_argument('--view_fusion_method', type=str, default='weighted', 
                        choices=['weighted', 'residual'],
                        help='fusion method for KNN view: weighted (fixed weights), or residual (pure residual with gate)')
    parser.add_argument('--view_weights', type=float, default=None,
                        help='weight for KNN view in weighted fusion (original view weight = 1 - KNN weight)')
    parser.add_argument('--view_knn_k', type=int, default=10,
                        help='K for KNN graph used in KNN view fusion')
    parser.add_argument('--view_knn_metric', type=str, default='cosine', choices=['cosine','euclidean'],
                        help='distance metric for KNN graph used in KNN view fusion')
    parser.add_argument('--knn_no_dropout', action='store_true',
                        help='disable dropout in KNN view for cleaner graph structure learning')
    parser.add_argument('--view_gate_init', type=float, default=0.0,
                        help='initial value for residual gate weight (only for residual fusion method)')
    parser.add_argument('--view_direct_fusion', action='store_true',
                        help='enable direct fusion mode: fuse two views at raw feature layer without going through local/global modules')


    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_dropout', type=float, default=None)

    # stability: normalization on input features
    parser.add_argument('--input_norm', type=str, default='none', choices=['none', 'nodenorm', 'pairnorm'],
                        help='apply NodeNorm/PairNorm to input features')

    # stability: label smoothing
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing factor in [0,1) for cross-entropy')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=100, help='how often to print')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')


