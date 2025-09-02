import pandas as pd
import streamlit as st

def main():

    st.title(':rainbow[**:sparkles: Nintendo Video Game Sales :sparkles:**]')
    st.markdown(':stars: :rainbow[Let\'s read in our dataset first] :stars:')
    st.code('''df = pd.read_csv('video_games_sales.csv')
df = df.dropna()

df = df[df['publisher'] == 'Nintendo']
''')
    
    df = pd.read_csv('video_games_sales.csv')
    df = df.dropna()

    df = df[df['publisher'] == 'Nintendo']
    
    st.dataframe(df)

    st.divider()
    st.header(':dango: :rainbow[Let\'s visualize our data!] :dango:')

    grouped = df.groupby(by = 'platform')
    total_sum_by_console = grouped['global_sales'].sum()
    total_sales_df = pd.DataFrame({'global_sales': total_sum_by_console})

    st.markdown(':video_game: :rainbow[Global Sales in Millions by Console] :video_game:')
    st.bar_chart(total_sales_df, x_label = 'Console', y_label = 'Global Sales in Millions', color = '#9A5B5E')
    
    avg_by_console = grouped['global_sales'].mean()
    avg_df = pd.DataFrame({'avg': avg_by_console})

    st.markdown(':video_game: :rainbow[Average Sales in Millions by Console] :video_game:')
    st.bar_chart(avg_df, x_label = 'Console', y_label = 'Average Sales in Millions', color = '#9A5B5E')

    st.markdown('- The Wii is the most successful Nintendo console both in global sales and in average global sales.')
    st.markdown('- Although the DS rivals the Wii in global sales, it pales in average global sales, ' \
    'suggesting that there were a few very very successful games on the DS, but for the Wii, Nintendo was releasing smash hit after smash hit.')
    st.markdown('- Although the 3DS and WiiU seem to be less successful, it is also tough to perform well after a huge global success such as the Wii.')
    st.markdown('- Also, being the most recent Nintendo consoles, their sales are ' \
    'probably still climbing steadily as newer games continue to be released on such systems.')

    st.divider()

    st.header(':coffee: :rainbow[Statistical Testing!] :coffee:')
    st.subheader(':rainbow-background[Hypothesis Test]')
    st.markdown('Is the difference between the average global sales of the Wii different from the average global sales of the NES?')
    st.markdown('$H_0: \mu_1 - \mu_2 = 0$ where $\mu_1$ is the average global sales of the Wii and $\mu_2$ is the average global sales of the NES.')
    st.markdown('$H_a: \mu_1 - \mu_2 \\neq 0$')

    st.markdown(':rainbow-background[Conditions!]')

    st.code('''wii = df[df['platform'] == 'Wii']
nes = df[df['platform'] == 'NES']

wii_mean = wii['global_sales'].mean()
wii_stdev = wii['global_sales'].std()
wii_sample_size = len(wii)
nes_mean = nes['global_sales'].mean()
nes_stdev = nes['global_sales'].std()
nes_sample_size = len(nes)''')
    
    st.markdown('''Wii mean:  4.819012345679012\n
Wii standard deviation:  11.608786392743458\n
Wii sample size:  81\n
NES mean:  4.088222222222222\n
NES standard deviation:  7.226626677178858\n
NES sample size:  45''')
    st.markdown('\n')

    st.markdown(':rainbow-background[Conditions!]')
    st.markdown('1. Sample of Wii games is randomly selected.')
    st.markdown('2. $n_{Wii}$ < 10% of all Wii games.')
    st.markdown('3. Sample of NES games is randomly selected.')
    st.markdown('4. $n_{NES}$ < 10% of all NES games.')
    st.markdown('5. $n_{Wii}$ = 84 > 30.')
    st.markdown('6. $n_{NES}$ = 45 > 30.')
    st.markdown('7. The two samples are independent of each other.')

    st.code('''wii_mean = wii['global_sales'].mean()
wii_stdev = wii['global_sales'].std()
wii_sample_size = len(wii)
nes_mean = nes['global_sales'].mean()
nes_stdev = nes['global_sales'].std()
nes_sample_size = len(nes)

point_estimate = wii_mean - nes_mean
standard_error = np.sqrt(((wii_stdev ** 2) / wii_sample_size) + ((nes_stdev ** 2) / nes_sample_size))
test_statistic = point_estimate / standard_error

degrees_freedom = min(wii_sample_size - 1, nes_sample_size - 1)

p_val = 2 * (1 - t.cdf(np.abs(test_statistic), df = degrees_freedom))''')
    
    st.markdown(':rainbow-background[p-value:] :rainbow[0.6657976899155496]')
    st.markdown('Using an $\\alpha$ value of 0.05, our p-value is way higher. ' \
    'Because our p-value is super high, we fail to reject the null hypothesis, ' \
    'so we do not have statistically significant evidence to suggest that there is a difference between the ' \
    'average global sales of Wii games and the average global sales of NES games.')

    st.subheader(':rainbow-background[Confidence Interval]')

    st.code('''t_star = t.ppf(0.995, df = degrees_freedom)
lower_bound = point_estimate - t_star * standard_error
upper_bound = point_estimate + t_star * standard_error''')

    st.markdown(':rainbow-background[Confidence Interval:] :rainbow[(-3.7937502417844424, 5.255330488698023)]')
    st.markdown('Since 0 is included within the bounds of the confidence interval, ' \
    'we do not have statistically significant evidence to suggest that there is a difference between the average ' \
    'global sales of Wii games and the average global sales of NES games.')

    st.divider()
    st.header(':city_sunrise: :rainbow[Insights From Statistical Tests] :city_sunrise:')
    st.markdown('1. We do not have enough evidence to suggest that the mean global sales between the Wii and NES are different.')
    st.markdown('2. In order to maximize profit, Nintendo should look to these systems to evaluate what made them so successful.')
    st.markdown('For example, we can look at the games that launched with the console release. If these games were well received, ' \
    'perhaps people would be more receptive to buying more games on these consoles since people already have the idea that games on this console are "good"')
    st.markdown(' We can also study what made these consoles different from the others, were they family-oriented? Were they individual-oriented?')
    st.markdown('These systems being marketed towards a family audience would make sense to why they were so successful, ' \
    'Nintendo is a family-centric business, which would help explain why these systems did better than the consoles that were more geared towards individuals, ' \
    'such as the DS, GB, and 3DS')
    st.markdown('Perhaps the gap in time is also significant here, the NES was released way before the Wii, ' \
    'almost as if the Wii is the NES\'s generational successor, perhaps Nintendo can take advantage of the next generation\'s "nostalgia"')
    st.markdown('To clarify, since the Wii was so successful, perhaps Nintendo can wait until people who grew up playing games on the Wii ' \
    'grew up to be active consumers, then release "nostalgic" games that are reminiscent of the Wii, ultimately using nostalgia to drive sales')

    st.divider()
    st.header(':bear: :rainbow[So... What Now? How Do We Determine What to Release to Maximize Sales?] :bear:')
    st.markdown(':high_heel: Let\'s categorize the games in this dataset as "high" risk or "low" risk depending on if they sold above or below the median.' \
    'Games that sold more than the median would be considered low risk whereas games that sold less than the median would be considered' \
    'high risk. :high_heel:')
    st.code('''median = df['global_sales'].median()

def categorize_risk(sales):
    if sales >= median:
        return 'low'
    else:
        return 'high'

df['risk'] = df['global_sales'].apply(categorize_risk)''')
 
    median = df['global_sales'].median()

    def categorize_risk(sales):
        if sales >= median:
            return 'low'
        else:
            return 'high'

    df['risk'] = df['global_sales'].apply(categorize_risk)

    st.markdown(':revolving_hearts: :rainbow[Let\'s see the new dataset with the risk column added!] :revolving_hearts:')
    st.dataframe(df)

    st.markdown('Let\'s also get creative with the data! We can group up all of the years by :rainbow-background[decade] so 1980\'s, 1990\'s, 2000\'s, and 2010\'s.')
    st.code('''def categorize_decade(year):
    if str(year // 10)[1] == '9':
        string = '19'
        return string + f'{int(year % 100):02d}'[0] + '0\'s'
    elif str(year // 10)[1] == '0':
        string = '20'
        return string + f'{int(year % 100):02d}'[0] + '0\'s' ''')

    def categorize_decade(year):
        if str(year // 10)[1] == '9':
            string = '19'
            return string + f'{int(year % 100):02d}'[0] + '0\'s'
        elif str(year // 10)[1] == '0':
            string = '20'
            return string + f'{int(year % 100):02d}'[0] + '0\'s'

    st.markdown('We can also :rainbow-background[categorize] our platforms as "handheld" or "family."')
    st.code('''handhelds = ['GB', 'GBA', 'DS', '3DS']
family = ['Wii', 'NES', 'SNES', 'N64', 'GC', 'WiiU']
def categorize_platform(platform):
    if platform in handhelds:
        return 'handheld'
    elif platform in family:
        return 'family' ''')
    
    handhelds = ['GB', 'GBA', 'DS', '3DS']
    family = ['Wii', 'NES', 'SNES', 'N64', 'GC', 'WiiU']

    def categorize_platform(platform):
        if platform in handhelds:
            return 'handheld'
        elif platform in family:
            return 'family'
    
    st.markdown('To more accurately depict the impact of each region on a game\'s sales, we can take their :rainbow-background[proportion] of each'
    'game\'s sales.')
    st.code('''df['na_share'] = df['na_sales'] / df['global_sales']
df['jp_share'] = df['jp_sales'] / df['global_sales']
df['eu_share'] = df['eu_sales'] / df['global_sales']''')
    
    df['decade'] = df['year'].apply(categorize_decade)
    df['console_type'] = df['platform'].apply(categorize_platform)
    df['na_share'] = df['na_sales'] / df['global_sales']
    df['jp_share'] = df['jp_sales'] / df['global_sales']
    df['eu_share'] = df['eu_sales'] / df['global_sales']
    
    st.markdown('Because some of the columns we want to use as labels are categorical, let\'s preprocess these columns'
    ' to make them usable for our machine learning model!')
    st.code('''ct = make_column_transformer(
    (OneHotEncoder(sparse_output = False), ['genre', 'platform', 'console_type', 'decade']),
    (OrdinalEncoder(), ['risk']),
    remainder = 'passthrough'
)

ct.set_output(transform = 'pandas')
df = ct.fit_transform(df)''')
    
    st.divider()
    st.header(':evergreen_tree: :rainbow[Building a RandomForestClassifier!] :evergreen_tree:')
    
    st.markdown('Now let\'s create our machine learning model and run 100 trials testing for accuracy so we can get a more'
    ' accurate depiction of the model\'s performance.')
    st.code('''trials = []

X = df.iloc[ : , 0:28]
X = pd.concat([X, df.iloc[ : , 37 : ]], axis = 1)
y = df['ordinalencoder__risk']

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    rf = RandomForestClassifier(n_estimators = 1000,
                                criterion = 'entropy',
                                min_samples_split = 40,
                                max_depth = 15,
                                class_weight = 'balanced'
                                )
    rf.fit(X_train, y_train)

    feature_importances = dict(zip(X.columns, rf.feature_importances_))
    feature_importances['score'] = rf.score(X_test, y_test)
    trials.append(feature_importances)

trial_df = pd.DataFrame(trials)''')
    
    st.markdown('Let\'s visualize the accuracy scores of all of these trials!')
    
    st.markdown('Let\'s see feature importances!')
    st.code('''accuracies = []
for col in trial_df.columns:
    accuracies.append((trial_df[col].median(), col))
accuracies.sort()
accuracies''')
    st.markdown('''[(np.float64(0.002477226202560072),\n'onehotencoder__genre_Fighting'),\n
 (np.float64(0.002763635267480733), 'onehotencoder__platform_3DS'),\n
 (np.float64(0.0027839216820237196), 'onehotencoder__genre_Shooter'),\n
 (np.float64(0.0032524263270429722), 'onehotencoder__platform_GC'),\n
 (np.float64(0.003433179291869642), 'onehotencoder__genre_Simulation'),\n
 (np.float64(0.004087634571707736), 'onehotencoder__genre_Action'),\n
 (np.float64(0.004286310581575779), 'onehotencoder__genre_Racing'),\n
 (np.float64(0.004817116645750083), 'onehotencoder__platform_Wii'),\n
 (np.float64(0.00509483583687508), 'onehotencoder__genre_Sports'),\n
 (np.float64(0.005711846513247973), 'onehotencoder__genre_Role-Playing'),\n
 (np.float64(0.006320453701574002), 'onehotencoder__platform_WiiU'),\n
 (np.float64(0.006450811144499793), 'onehotencoder__platform_N64'),\n
 (np.float64(0.006453132192272985), 'onehotencoder__genre_Puzzle'),\n
 (np.float64(0.007627228233043952), 'onehotencoder__genre_Misc'),\n
 (np.float64(0.007917562620716124), 'onehotencoder__platform_DS'),\n
 (np.float64(0.008196290609382347), 'onehotencoder__platform_SNES'),\n
 (np.float64(0.008212128678716273), 'onehotencoder__console_type_family'),\n
 (np.float64(0.008423753329225543), 'onehotencoder__console_type_handheld'),\n
 (np.float64(0.00846943048146586), 'onehotencoder__genre_Strategy'),\n
 (np.float64(0.010545123657148311), 'onehotencoder__platform_GBA'),\n
 :blue[(np.float64(0.011563964127852103), 'onehotencoder__genre_Adventure')],\n
 (np.float64(0.012364522687918246), "onehotencoder__decade_2010's"),\n
 (np.float64(0.014451191918244038), "onehotencoder__decade_2000's"),\n
 :blue[(np.float64(0.01851232238936843), 'onehotencoder__genre_Platform')],\n
 (np.float64(0.023025857729514418), "onehotencoder__decade_1990's"),\n
 :green[(np.float64(0.0294937741701021), 'onehotencoder__platform_GB')],\n
 :green[(np.float64(0.03107421212268413), 'onehotencoder__platform_NES')],\n
 :violet[(np.float64(0.03607018709915095), "onehotencoder__decade_1980's")],\n
 :rainbow[(np.float64(0.20932597469659187), 'remainder__eu_share')],\n
 :rainbow[(np.float64(0.23791377835041821), 'remainder__jp_share')],\n
 :rainbow[(np.float64(0.2516844363223184), 'remainder__na_share')],\n
 :red[(np.float64(0.8214285714285714), 'score')]]''')
    
    st.divider()
    st.header(':cherry_blossom: :rainbow[Insights From Our Model] :cherry_blossom:')
    st.markdown('It seems that :rainbow[na_share, jp_share, and eu_share (rainbow text)] are the most important features that carry the model\'s precision.'
    ' However, it is also really interesting to see that many "older" categories, such as the :violet[1980\'s decade (purple text)], ' \
    ' :green[the NES, and the GB (green text)]'
    ' are pretty high up on the feature importances. Of course, we cannot rule out the power of time and nostalgia because these games have been "out"'
    ' for the longest, so there is more time for them to be sold compared to games released on the newest systems. In terms of genre, the top two being'
    ' :blue[platform and adventure (blue text)] signal the Mario and Legend of Zelda franchises respectively, two of Nintendo\'s largest'
    ' franchises that pretty consistently sell well. This makes sense because if those two franchises sell well, they would be considered to be'
    ' pretty "safe" releases in terms of sales and profit, so it would increase the importance of these two genres for the model.')

    st.divider()
    st.header(':dart: :rainbow[What EXACTLY Should We Release?] :dart:')
    st.markdown('1. Since North American sales was the most important feature in the machine learning model, perhaps we can cater to North American cultures' \
    ' through easter eggs, unlockables, etc')
    st.markdown('We can also look to release games that are part of popular franchises in North America because certain genres are more popular in some'
    ' regions than others.')
    st.markdown('Of course, it is important not to neglect the other markets because they are also a pretty large feature in the model.')
    st.markdown('2. We can leverage nostalgia from older systems in order to drive sales, almost like a full circle moment in the marketing.')
    st.markdown('In other words, we should draw inspiration from the 80\'s but put a modern twist.')
    st.markdown('For example, say we examine a Mario game from the NES, we can take elements from that game that maybe we scrapped in future '
    'installments and bring them back. This will leverage the stronger features of the 1980\'s decade and the platforming genre.')



if __name__ == '__main__':
    main()