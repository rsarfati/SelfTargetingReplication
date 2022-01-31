# % Table 8. Estimated Parameter Values for the Model, and
# % Online Appendix Table C.15. Estimated Parameters Values for the Model (Discount Rate Robustness Check)
# % Estimates
# run estimation_1.m;
#     % Input:  MATLAB_Input.csv
#     % Output: MATLAB_Estimates_main_d**.csv
#     %    ** corresponds to the value of the annual discount factor: d=0.82 = 1/irate, d=0.50 or d=0.95
#

# % Standard errors
# %run bootstrapping.m
#     % run three times, changing line 19: danual  = 1/irate; danual = 0.50; and danual = 0.95
#     % Input:  MATLAB_Input.csv
#     % Output: bootstrap_**\bootstrap_[something]_d**.csv files
#
# % print standard deviations
# boot_estimates_82 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_82\bootstrap_estimates_82.csv']);
# boot_estimates_50 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_50\bootstrap_estimates_50.csv']);
# boot_estimates_95 = csvread([t2folder 'data\codeddata\Matlab\bootstrap_95\bootstrap_estimates_95.csv']);
#
# boot_estimates_82_std = std(boot_estimates_82(:,[3,6,9,12,15]));
# boot_estimates_50_std = std(boot_estimates_50(:,[3,6,9,12,15]));
# boot_estimates_95_std = std(boot_estimates_95(:,[3,6,9,12,15]));
#
# display(['Boostrap standard deviation for delta = 0.50 is ' num2str(boot_estimates_50_std,5)]);
# display(['Boostrap standard deviation for delta = 0.82 is ' num2str(boot_estimates_82_std,5)]);
# display(['Boostrap standard deviation for delta = 0.95 is ' num2str(boot_estimates_95_std,5)]);
#
#
#
# % counterfactual showup hat used in Tables 9 and 10, and
# % Online Appendix Table C.18.
# run counterfactuals_1.m
#     % Input:  MATLAB_Input.csv,
#     %         MATLAB_Input_small_sample.csv
#     % Output: MATLAB_showuphat_small_sample.csv
#     %         MATLAB_showuphat.csv
