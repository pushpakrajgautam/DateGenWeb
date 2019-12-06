from django import forms

class ProfileForm(forms.Form):
    age = forms.IntegerField(label="Age", min_value=0, max_value=70, initial=21)
    body_type = forms.ChoiceField(label="Body Type", 
                                  choices=[
                                    ('rather not say', 'Rather Not Say'),
                                    ('full figured', 'Full Figured'),
                                    ('overweight', 'Overweight'),
                                    ('thin', 'Thin'),
                                    ('curvy', 'Curvy'),
                                    ('athletic', 'Athletic'),
                                    ('used up', 'Used Up'),
                                    ('skinny', 'Skinny'),
                                    ('a little extra', 'A Little Extra'),
                                    ('jacked', 'Jacked'),
                                    ('average', 'Average'),
                                    ('fit', 'Fit'),
                                  ])
    drinks = forms.ChoiceField(label="Drinks", 
                               choices=[
                                ('not at all', 'Not At All'), 
                                ('very often', 'Very Often'), 
                                ('socially', 'Socially'), 
                                ('often', 'Often'), 
                                ('desperately', 'Desperately'), 
                                ('rarely', 'Rarely'),
                               ])
    education = forms.ChoiceField(label="Education",
                                  choices=[
                                    ('college/university', 'College/University'),
                                    ('space camp', 'Space Camp'),
                                    ('law school', 'Law School'),
                                    ('high school', 'High School'),
                                    ('med school', 'Med School'),
                                    ('masters program', 'Masters Program'),
                                    ('ph.d program', 'PhD Program'),
                                  ])
    job = forms.ChoiceField(label="Job",
                            choices=[
                                ('student', 'Student'), 
                                ('medicine / health', 'Medicine'), 
                                ('political / government', 'Government'), 
                                ('entertainment / media', 'Entertainment'), 
                                ('executive / management', 'Management'),
                                ('hospitality / travel', 'Travel'),
                                ('law / legal services', 'Law'),
                                ('artistic / musical / writer', 'Artist'), 
                                ('sales / marketing / biz dev', 'Sales'),
                                ('banking / financial / real estate', 'Finance'), 
                                ('education / academia', 'Academia'),
                                ('military', 'Military'),
                                ('computer / hardware / software', 'Tech'), 
                                ('retired', 'Retired'),
                                ('unemployed', 'Unemployed'),
                                ('other', 'Other'),
                                ('rather not say', 'Rather Not Say'), 
                            ])
    orientation = forms.ChoiceField(label="Orientation",
                                    choices=[
                                        ('gay', 'Gay'),
                                        ('straight', 'Straight'),
                                        ('bisexual', 'Bisexual'),
                                    ])
    pets = forms.ChoiceField(label="Pets",
                             choices=[
                                ('has dogs', 'Dogs'),
                                ('has cats', 'Cats'),
                                ('has dogs and has cats', 'Both'),
                                ('dislikes dogs and dislikes cats', 'Neither'),
                             ])
    sex = forms.ChoiceField(label="Sex",
                            choices=[
                                ('f', 'Female'),
                                ('m', 'Male'),
                            ])
    smokes = forms.ChoiceField(label="Smokes",
                               choices=[
                                   ('no', 'No'),
                                   ('yes', 'Yes'),
                                   ('sometimes', 'Sometimes'), 
                                   ('when drinking', 'When Drinking'),
                                   ('trying to quit', 'Trying to Quit'),
                               ])
    status = forms.ChoiceField(label="Status",
                               choices=[
                                   ('single', 'Single'),
                                   ('married', 'Married'), 
                                   ('available', 'Available'),
                                   ('seeing someone', 'Seeing Someone'),
                                   ('unknown', 'Unknown'),
                               ])
    temp = forms.FloatField(label="Temperature", initial=1.5)