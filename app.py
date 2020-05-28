
#----- DATA IMPORT AND CLEANING
import numpy as np
import pandas as pd

#----- APP SETUP
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from datetime import timedelta

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)


# app layout
server = app.server
app.title='Covid-19 Dashboard'

REMOTE_DATAFILE = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"


def read_coviddata(filename=REMOTE_DATAFILE,popThr=200000):
    "reads and cleans covid data. Countries with population below popThr are excluded"

    dfw = pd.read_excel(filename,parse_dates = ['dateRep'])
    # get last update date 
    global last_date 
    last_date = dfw['dateRep'].iloc[0]
    # delete weird cases
    dfw.drop(dfw[dfw.countriesAndTerritories=='Cases_on_an_international_conveyance_Japan'].index,inplace=True)
    # sort by country and date
    dfw = dfw.sort_values(by=['countriesAndTerritories','dateRep'])
    # reset ix
    dfw=dfw.reset_index(drop=True)

    # col renaming
    dfw = dfw.rename(columns={'dateRep':'date',
                              'countriesAndTerritories':'country',
                              'countryterritoryCode':'countryCode',
                              'popData2018':'population',
                              'continentExp':'continent'})
    # drop cols
    dfw= dfw.drop(columns=['day','month','year'])
    # remove countries under pop threshold
    dfw = dfw[dfw.population>popThr]
    # remove leading zeros by country
    dfw=dfw.groupby('country',as_index=False).apply(remove_leading_zeros)
    # reset ix
    dfw=dfw.reset_index(drop=True)
    # add daycount
    dfw['daycount'] = (dfw['date']-dfw['date'].min()).apply(lambda x: x.days)
    #
    return dfw

def remove_leading_zeros(sub):
    "removes the initial date points where cases and deaths are zero"
    
    zeroFlag = (sub.cases==0) & (sub.deaths==0)
    if not zeroFlag.iloc[0]:
        return sub
    #indices where flag is true
    ix0 = zeroFlag[zeroFlag].index
    # diff of indices
    delta_ix0 = np.diff(ix0)
    # jump indices
    jump_ix=np.where(delta_ix0>1)
    # if no jump, there is only one section with zeros
    if list(jump_ix[0])==[]:
        first_discontinuity_ix= -1
    else:
        first_discontinuity_ix = jump_ix[0][0]
    #
    out = sub.loc[ix0[first_discontinuity_ix]+1:]
    #
    return out

#---- MATH FUNCTIONS
def lnfit(x,y):
    "Fits linearly the log(y) - natural log"
    # remove points where log does not exist
    x=x[y>0]
    y=y[y>0]
    f = np.polyfit(x, np.log(y),1)
    return f

def lnval(f, inp):
    "Evaluate exp**fit(x)"
    return np.exp(np.polyval(f,inp))

import warnings
def relative_growth(daily_series):
    "Calculates the relative growth starting from a daily series"
    cum_series = daily_series.cumsum()
    # suppress warnings as you might have division by zero
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        start = np.array([np.nan])
        rv = daily_series.values[1:]/cum_series.values[:-1] # relative variation
        alldata = np.concatenate([start,rv])
    #
    return pd.Series(alldata,index=daily_series.index)


def adaptive_moving_average(y, window=7):
    """centered moving average with window reduction at the extremities.
    Window size must be uneven"""
    assert window%2 == 1
    # standard moving average
    yout = y.rolling(window=window,center=True).mean()
    # extremities
    nextremity = int((window-1)/2)
    for k in range(nextremity):
        yout.iloc[k] = y.iloc[0:2*k+1].mean()
        yout.iloc[-(k+1)] = y.iloc[-(2*k+1):].mean()
    #
    return yout


#---- SIGNAL EXTRACTION

def extract_signal(dfw,country,yvalue,signaltype,popFlag,maFlag,extrapFlag,
                   extrap_last_ndays=15,extrap_ndays=60,refPop=1e4):
    """Extracts signal for given country considering signal type (cases|deaths) and
    optional flags like population normalisation, moving average and extrapolation"""

    sub = dfw[dfw.country==country] 
    x = sub['date']
    y = sub[yvalue]
    # signal type
    assert signaltype in ['daily','relative growth','cumulative']
    if signaltype == 'cumulative':
        y = np.cumsum(y)
    if signaltype == 'relative growth':
        y = relative_growth(y)
    # popflag
    if popFlag==['Norm'] and signaltype!='relative growth': # do not scale by pop if relgrowth
        y = y/sub['population'].iloc[0]*refPop
    # moving average
    if maFlag==['MA']:
        y = adaptive_moving_average(y)
    # extrapolation
    if extrapFlag==['Extrap']:
        # calculate relative growth from daily series
        relgrowth = relative_growth(sub[yvalue])
        # extract last data points
        finaldaycounts = sub['daycount'].iloc[-extrap_last_ndays:]
        finalgrowths = relgrowth.iloc[-extrap_last_ndays:]
        # try ln interpolation
        try:
            resfit = lnfit(finaldaycounts,finalgrowths)
        except:
            resfit = []
        #
        # extrapolate only if fit exists and has negative slope
        if type(resfit)==np.ndarray:
            lastdaycount = sub['daycount'].iloc[-1]
            # extra days
            extra_daycount = np.arange(lastdaycount+1,lastdaycount+extrap_ndays+1)
            newdays = pd.date_range(dfw.date.iloc[-1]+timedelta(days=1)
                                    ,dfw.date.iloc[-1]+timedelta(days=extrap_ndays))
            # concatenate x
            x= pd.concat((x,pd.Series(newdays)),axis=0)
            #
            if resfit[0]<0:
                # total values at last collection date
                total = sub[yvalue].sum()
                # extrapolate with negative slope
                extra_growth = lnval(resfit,extra_daycount)
                extra_cumulative = total*np.cumprod(1+extra_growth)
                extra_daily = np.concatenate( (np.array([extra_cumulative[0]-total]),np.diff(extra_cumulative)) )
                #
                if signaltype=='daily':
                    yextra = extra_daily
                elif signaltype=='cumulative':
                    yextra = extra_cumulative
                elif signaltype=='relative growth':
                    yextra = extra_growth
                # pop normalise
                if popFlag==['Norm'] and not signaltype=='relative growth':
                    yextra = yextra/sub.population.iloc[0]*refPop
            else:
                # with positive slope just report nan's
                yextra = np.zeros((len(newdays),))
                yextra[:]= np.nan
                                           
            # concatenate y
            y= np.concatenate((y,yextra))
            #
    return x,y       



# read data file
dfw = read_coviddata()
# countries
countries = dfw.country.unique()

#
app.layout = html.Div(children=[
    html.H2('Covid19 World Status Analysis',style={'textAlign':'center'}),
    html.H4(f'Last update : {last_date}' ,style={'textAlign':'center'}),
    html.H6("____________________________________________________________________",style={'textAlign':'center'}),


    html.Div(children=[
        html.Label('Countries'),
        dcc.Dropdown(id='country_selection',options=[{'label': x,'value':x} for x in countries], 
                    value=['Egypt'],multi=True),
        #
        html.Label('Y-values'),
        dcc.Dropdown(id='ycolumn',options=[ {'label': 'Cases','value':'cases'}, 
                                            {'label': 'Deaths','value':'deaths'} ],value='cases'),
        #
        html.Div(children=[
        html.Label('Signal type'),
        dcc.RadioItems(
                id='signal-type',
                options=[{'label': i, 'value': i.lower()} for i in ['Daily', 'Cumulative','Relative Growth']],
                value='daily',
                ),]),
        
        html.Label('Y-scale'),
        dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i.lower()} for i in ['Linear', 'Log']],
                value='linear',
                labelStyle={'display': 'inline-block'}),
        html.Label('Options'),
        #
        dcc.Checklist(  id='norm-flag',
                        options=[{'label':'Population Normalise','value':'Norm'}],value=[]),
        dcc.Checklist(  id='ma-flag',
                        options=[{'label':'Moving Average','value':'MA'}],value=[]),
        dcc.Checklist(  id='extrap-flag',
                        options=[{'label':'Time Extrapolation','value':'Extrap'}],value=[]),
        html.Br(),
        # dcc.Link('Covid-19 Chatbot', href='/covid-19-chatbot/app_chatbot.py'),
        dcc.Markdown(('[Github link](https://github.com/kareem-desouky22/Corona-Dashboard-Chatbot)')),
        dcc.Markdown(('Developed by Wael abouelwafaa & Kareem Desouky '
                  ' using [DASH](https://plot.ly/dash/)')),

        ]
        ,style={'width': '22%', 'display': 'inline-block'}
        ),

    dcc.Graph(id='timeplot',style={'width':'74%','float':'right','margin':'22px' })
])

# APP callback
@app.callback(
    Output('timeplot', 'figure'),
    [Input('country_selection', 'value'),
     Input('ycolumn', 'value'),
     Input('signal-type','value'),
     Input('yaxis-type', 'value'),
     Input('norm-flag','value'),
     Input('ma-flag','value'),
     Input('extrap-flag','value')])
def update_graph(countries,yvalue,signaltype,yaxistype,popFlag,maFlag,extrapFlag):
    
    # traces
    traces = []
    for country in countries:
        
        x, y = extract_signal(dfw,country,yvalue,signaltype,popFlag,maFlag,extrapFlag)
        traces.append( dict(x=x, y=y,mode='lines',name=country))
    
    # layout
    if popFlag == ['Norm'] and signaltype!='relative growth':
        normLab = ' [per 10.000 people]'
    else:
        normLab = ''
    #
    if maFlag ==['MA']:
        maLab= ' (Moving Average)'
    else:
        maLab= ''
    #
    ytitle = signaltype.capitalize() + normLab + maLab
    # 
    layout = dict(
                    xaxis={'title':'date'},
                    yaxis={'type':yaxistype,'title':ytitle},
                    margin={'l': 80, 'b': 40, 't': 50, 'r': 10},
                    title= {'text':yvalue.capitalize(),'font':{'size':25}},
                    )
    if extrapFlag == ['Extrap']:
        lastcollecteddate = dfw['date'].max()
        lastplotdate = x.max()
        layout['shapes'] =  [  dict(
                                type="rect",
                                # x-reference is assigned to the x-values
                                xref="x",
                                # y-reference is assigned to the plot paper [0,1]
                                yref="paper",
                                x0=lastcollecteddate,
                                y0=0,
                                x1=lastplotdate,
                                y1=1.05,
                                fillcolor="LightGreen",
                                opacity=0.5,
                                layer="below",
                                line={'width':0},)
        ]
        layout['annotations'] = [  dict(
                                        text=' Projection',
                                        xanchor='left',
                                        showarrow=False,
                                        # x-reference is assigned to the x-values
                                        xref="x",
                                        # y-reference is assigned to the plot paper [0,1]
                                        yref="paper",
                                        x=lastcollecteddate,
                                        y=1.05,
                                        font= {'size':18})
        
        ]
    # 
    return {'data':traces,
            'layout':layout}





if __name__ == '__main__':
    app.run_server(debug=False)