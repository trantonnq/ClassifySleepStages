import time

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=FutureWarning)


from source import utils
from source.analysis.classification.classifier_summary_builder import SleepWakeClassifierSummaryBuilder, \
    ThreeClassClassifierSummaryBuilder, FourClassClassifierSummaryBuilder, FiveClassClassifierSummaryBuilder
from source.analysis.figures.curve_plot_builder import CurvePlotBuilder
from source.analysis.figures.performance_plot_builder import PerformancePlotBuilder
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.tables.table_builder import TableBuilder
from source.constants import Constants


def figures_leave_one_out_sleep_wake_performance():
    attributed_classifier = AttributedClassifier(name='Neural Net',
                                                 classifier=MLPClassifier(activation='relu',
                                                                          hidden_layer_sizes=(15, 15, 15),
                                                                          max_iter=1000, alpha=0.01, solver='lbfgs'))

    feature_sets = [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]

    if Constants.VERBOSE:
        print('Running ' + attributed_classifier.name + '...')
    classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)
    PerformancePlotBuilder.make_histogram_with_thresholds(classifier_summary)
    PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary)


def figures_leave_one_out_three_class_performance():
    attributed_classifier = AttributedClassifier(name='Neural Net',
                                                 classifier=MLPClassifier(activation='relu',
                                                                          hidden_layer_sizes=(15, 15, 15),
                                                                          max_iter=1000, alpha=0.01, solver='lbfgs'))

    feature_sets = utils.get_base_feature_sets()

    if Constants.VERBOSE:
        print('Running ' + attributed_classifier.name + '...')
    classifier_summary = ThreeClassClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)
    PerformancePlotBuilder.make_bland_altman(classifier_summary)

def figures_leave_one_out_five_class_performance():
    attributed_classifier = AttributedClassifier(name='Neural Net',
                                                 classifier=MLPClassifier(activation='relu',
                                                                          hidden_layer_sizes=(15, 15, 15),
                                                                          max_iter=1000, alpha=0.01, solver='lbfgs'))

    feature_sets = utils.get_base_feature_sets()

    if Constants.VERBOSE:
        print('Running ' + attributed_classifier.name + '...')
    classifier_summary = FiveClassClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)
    PerformancePlotBuilder.make_bland_altman(classifier_summary)


def figure_leave_one_out_roc_and_pr():
    classifiers = utils.get_classifiers()
    feature_sets = utils.get_base_feature_sets()

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_one_out(attributed_classifier, feature_sets)

        CurvePlotBuilder.make_roc_sw(classifier_summary)
        CurvePlotBuilder.make_pr_sw(classifier_summary)
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, len(SubjectBuilder.get_all_subject_ids()), '_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, len(SubjectBuilder.get_all_subject_ids()), '_sw_roc')


def figures_mc_sleep_wake():
    classifiers = utils.get_classifiers()
    
    feature_sets = utils.get_base_feature_sets()
    trial_count = 20

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                 trial_count)

        CurvePlotBuilder.make_roc_sw(classifier_summary)
        CurvePlotBuilder.make_pr_sw(classifier_summary)
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_sw_roc')

# Testing
def figures_mc_three_class():
    #classifiers = utils.get_classifiers()
    classifiers =  [
                    # AttributedClassifier(name='Random Forest',
                    #              classifier=RandomForestClassifier(n_estimators=100, max_features=1.0,
                    #                                                max_depth=10,
                    #                                                min_samples_split=10, min_samples_leaf=32,
                    #                                                bootstrap=True))
                    #, 
                    # AttributedClassifier(name='k-Nearest Neighbors',
                    #              classifier=KNeighborsClassifier(weights='distance')),
                    # AttributedClassifier(name='Logistic Regression',
                    #              classifier=LogisticRegression(penalty='l1', solver='liblinear', verbose=0,
                    #                                            multi_class='auto')),
                    AttributedClassifier(name='Neural Net',
                                 classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                                                          max_iter=2000, alpha=0.01, solver='adam', verbose=False,
                                                          n_iter_no_change=20))
                    ]
    #feature_sets = utils.get_base_feature_sets()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]
    trial_count = 20

    three_class_performance_summaries = []
    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                  trial_count)

        CurvePlotBuilder.make_roc_one_vs_rest(classifier_summary)
    #     three_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary)

    #     classifier_summary.performance_dictionary = three_class_performance_dictionary
    #     three_class_performance_summaries.append(classifier_summary)

    # TableBuilder.print_table_three_class(three_class_performance_summaries)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_three_class_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_rem_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_nrem_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_ovr_wake_roc')

def figures_mc_four_class():
    #classifiers = utils.get_classifiers()
    classifiers =  [
                    AttributedClassifier(name='k-Nearest Neighbors',
                                 classifier=KNeighborsClassifier(weights='distance',algorithm='kd_tree')),

                    AttributedClassifier(name='Logistic Regression',
                                 classifier=LogisticRegression(penalty='l1', solver='liblinear', verbose=0,
                                                                multi_class='auto'))
                    ,
                    AttributedClassifier(name='Random Forest',
                                 classifier=RandomForestClassifier(n_estimators=100, max_features=1.0,
                                                                   max_depth=10,
                                                                   min_samples_split=10, min_samples_leaf=32,
                                                                   bootstrap=True))
                   
                    ,
                    AttributedClassifier(name='AdaBoostClassifier',
                                 classifier=AdaBoostClassifier())
                    #,
                    # # AttributedClassifier(name='Neural Net',
                    #              classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                    #                                       max_iter=2000, alpha=0.01, solver='adam', verbose=False,
                    #                                       n_iter_no_change=20))
                    ]

    #feature_sets = utils.get_base_feature_sets()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]
    trial_count = 5

    four_class_performance_summaries = []

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = FourClassClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                  trial_count)

        CurvePlotBuilder.make_roc_one_vs_rest_four_class(classifier_summary)
        #five_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary)

        #classifier_summary.performance_dictionary = five_class_performance_dictionary
        #four_class_performance_summaries.append(classifier_summary)

    #TableBuilder.print_table_three_class(four_class_performance_summaries)
    #CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_three_class_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '4_stages_ovr_WAKE_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '4_stages_ovr_N1_N2_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '4_stages_ovr_N3_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '4_stages_ovr_REM_roc')


def figures_mc_five_class():
    #classifiers = utils.get_classifiers()
    classifiers =  [
                    AttributedClassifier(name='k-Nearest Neighbors',
                                 classifier=KNeighborsClassifier(weights='distance',algorithm='kd_tree')),

                    AttributedClassifier(name='Logistic Regression',
                                 classifier=LogisticRegression(penalty='l1', solver='liblinear', verbose=0,
                                                                multi_class='auto'))
                    ,
                    AttributedClassifier(name='Random Forest',
                                 classifier=RandomForestClassifier(n_estimators=100, max_features=1.0,
                                                                    max_depth=10,
                                                                    min_samples_split=10, min_samples_leaf=32,
                                                                    bootstrap=True))
                                       ,
                    AttributedClassifier(name='AdaBoostClassifier',
                                  classifier=AdaBoostClassifier())
                    # ,
                    # AttributedClassifier(name='Neural Net',
                    #              classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
                    #                                       max_iter=2000, alpha=0.01, solver='adam', verbose=False,
                    #                                       n_iter_no_change=20))
                    ]

    #feature_sets = utils.get_base_feature_sets()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]
    trial_count = 5

    five_class_performance_summaries = []

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = FiveClassClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                  trial_count)

        CurvePlotBuilder.make_roc_one_vs_rest_five_class(classifier_summary)
        #five_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary)

        #classifier_summary.performance_dictionary = five_class_performance_dictionary
        #five_class_performance_summaries.append(classifier_summary)

    #TableBuilder.print_table_three_class(five_class_performance_summaries)
    # CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_three_class_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '5_stages_ovr_WAKE_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '5_stages_ovr_N1_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '5_stages_ovr_N2_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '5_stages_ovr_N3_roc')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '5_stages_ovr_REM_roc')


def figures_mesa_sleep_wake():
    classifiers = utils.get_classifiers()
    # Uncomment to just use MLP:
    # classifiers = [AttributedClassifier(name='Neural Net',
    #                                     classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
    #                                                              max_iter=1000, alpha=0.01, solver='lbfgs'))]

    feature_sets = utils.get_base_feature_sets()

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_mesa(attributed_classifier, feature_sets)
        CurvePlotBuilder.make_roc_sw(classifier_summary, '_mesa')
        CurvePlotBuilder.make_pr_sw(classifier_summary, '_mesa')
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_sw_roc')


def figures_mesa_three_class():
    classifiers = utils.get_classifiers()

    # Uncomment to just use MLP:
    # classifiers = [AttributedClassifier(name='Neural Net', classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
    #                                                            max_iter=1000, alpha=0.01, solver='lbfgs'))]

    feature_sets = utils.get_base_feature_sets()
    three_class_performance_summaries = []

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_mesa_leave_one_out(attributed_classifier,
                                                                                         feature_sets)
        PerformancePlotBuilder.make_bland_altman(classifier_summary, '_mesa')
        PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary, '_mesa')

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = ThreeClassClassifierSummaryBuilder.build_mesa_all_combined(attributed_classifier,
                                                                                        feature_sets)
        three_class_performance_dictionary = CurvePlotBuilder.make_three_class_roc(classifier_summary, '_mesa')
        classifier_summary.performance_dictionary = three_class_performance_dictionary
        three_class_performance_summaries.append(classifier_summary)
        CurvePlotBuilder.combine_sw_and_three_class_plots(attributed_classifier, 1, 'mesa')

    TableBuilder.print_table_three_class(three_class_performance_summaries)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_three_class_roc')

def figures_mesa_five_class():
    classifiers = utils.get_classifiers()

    # Uncomment to just use MLP:
    # classifiers = [AttributedClassifier(name='Neural Net', classifier=MLPClassifier(activation='relu', hidden_layer_sizes=(15, 15, 15),
    #                                                            max_iter=1000, alpha=0.01, solver='lbfgs'))]

    feature_sets = utils.get_base_feature_sets()
    five_class_performance_summaries = []

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = FiveClassClassifierSummaryBuilder.build_mesa_leave_one_out(attributed_classifier,
                                                                                         feature_sets)
        PerformancePlotBuilder.make_bland_altman(classifier_summary, '_mesa')
        PerformancePlotBuilder.make_single_threshold_histograms(classifier_summary, '_mesa')

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = FiveClassClassifierSummaryBuilder.build_mesa_all_combined(attributed_classifier,
                                                                                        feature_sets)
        five_class_performance_dictionary = CurvePlotBuilder.make_five_class_roc(classifier_summary, '_mesa')
        classifier_summary.performance_dictionary = five_class_performance_dictionary
        five_class_performance_summaries.append(classifier_summary)
        CurvePlotBuilder.combine_sw_and_five_class_plots(attributed_classifier, 1, 'mesa')

    TableBuilder.print_table_five_class(five_class_performance_summaries)
    CurvePlotBuilder.combine_plots_as_grid(classifiers, 1, '_mesa_five_class_roc')    

def figures_compare_time_based_features():
    classifiers = utils.get_classifiers()
    feature_sets = [[FeatureType.count, FeatureType.heart_rate],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.time],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.cosine],
                    [FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model]]

    trial_count = 50

    for attributed_classifier in classifiers:
        if Constants.VERBOSE:
            print('Running ' + attributed_classifier.name + '...')
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_monte_carlo(attributed_classifier, feature_sets,
                                                                                 trial_count)

        CurvePlotBuilder.make_roc_sw(classifier_summary, '_time_only')
        CurvePlotBuilder.make_pr_sw(classifier_summary, '_time_only')
        TableBuilder.print_table_sw(classifier_summary)

    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_time_only_sw_pr')
    CurvePlotBuilder.combine_plots_as_grid(classifiers, trial_count, '_time_only_sw_roc')

def main():
    start_time = time.time()
    #figure_leave_one_out_roc_and_pr()

    #figures_mc_sleep_wake()
    #figures_mc_three_class()
    figures_mc_five_class()
    figures_mc_four_class()

    #figures_leave_one_out_sleep_wake_performance()
    #figures_leave_one_out_three_class_performance()
    #figure_leave_one_out_roc_and_pr()

    #figures_mesa_sleep_wake()
    #figures_mesa_three_class()
    #figures_mesa_five_class()

    #figures_compare_time_based_features()
    end_time = time.time()

    print('Elapsed time to generate figure: ' + str((end_time - start_time) / 60) + ' minutes')

if __name__ == "__main__":
    main()
