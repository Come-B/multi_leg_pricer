# coding: utf-8

# TO LAUNCH, RUN THIS IN A COMMAND PROMPT/TERMINAL
# the first 2 lines depend on this file location on your computer (1st is useless if terminal is on the same partition)

# D:/
# cd Come/Documents/"Master 203"/"Python for finance (PY)"/Project  
# bokeh serve --show project.py  

import pandas as pd
from pandas.api.types import is_numeric_dtype
import bokeh
import numpy as np
import scipy.stats as stats
from arch import arch_model

import base64
from io import BytesIO

from bokeh.palettes import Spectral4
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Dropdown, CustomJS, Select, Spinner, Range1d, Toggle, \
 Title, Panel, Tabs, RadioButtonGroup, Button, CheckboxGroup, CheckboxButtonGroup, DataTable, TableColumn, FileInput, Div, PreText
from bokeh.plotting import figure


def BS_option_complete(nature, S, K, vol_annual, T, r_annual):
    r = r_annual/int(days_in_year_input.value)
    vol = vol_annual/np.sqrt(int(days_in_year_input.value))

    d_1 = (np.log(S/K) + (r + vol**2/2)*T)/(vol*np.sqrt(T))
    d_2 = d_1 - vol*np.sqrt(T)

	# Put and Call prices
    call_price = S*stats.norm.cdf(d_1) - K*np.exp(-r*T)*stats.norm.cdf(d_2)
    put_price = -S*stats.norm.cdf(-d_1) + K*np.exp(-r*T)*stats.norm.cdf(-d_2)

    delta_call = stats.norm.cdf(d_1)
    delta_put = stats.norm.cdf(d_1)-1

    gamma_call = 1/(vol*S*np.sqrt(T))*stats.norm.pdf(d_1)
    gamma_put = gamma_call

    theta_call = -vol*S/(2*np.sqrt(T))*stats.norm.pdf(d_1) - r*K*np.exp(-r*T)*stats.norm.cdf(d_2)
    theta_put = -vol*S/(2*np.sqrt(T))*stats.norm.pdf(d_1) + r*K*np.exp(-r*T)*stats.norm.cdf(-d_2)

    vega_call = S*np.sqrt(T)*stats.norm.pdf(d_1)
    vega_put = vega_call

    rho_call = T*K*np.exp(-r*T)*stats.norm.cdf(d_2)
    rho_put = -T*K*np.exp(-r*T)*stats.norm.cdf(-d_2)

    if(nature == 'call'):
        return (call_price, delta_call, gamma_call, theta_call, vega_call, rho_call)
    return (put_price, delta_put, gamma_put, theta_put, vega_put, rho_put)


def BS_call_complete(S, K, vol, T, r):
    return BS_option_complete('call', S, K,  vol, T, r)


def BS_put_complete(S, K, vol, T, r):
    return BS_option_complete('put', S, K, vol, T, r)

assets = {
         'Call option': {'payoff': (lambda S, K: np.maximum(0, S-K)), 'BS_analytical': BS_call_complete},
         'Put option': {'payoff': (lambda S, K: np.maximum(0, K-S)), 'BS_analytical': BS_put_complete},
         'Spot': {'payoff': (lambda S, K: S), 'BS_analytical': (lambda S, K, vol, T, r: (S, 1, 0,  0, 0, 0))},
         }


def BS_path(theta, S0, T, N_points):
    r_annual, vol_annual = theta
    r = r_annual/int(days_in_year_input.value)
    vol = vol_annual/np.sqrt(int(days_in_year_input.value))

    res = S0*np.ones(N_points+1)
    res[1:] = S0*np.cumprod(np.exp((r - vol**2/2) * T/N_points + vol * np.random.standard_normal(size=(N_points)) * np.sqrt(T/N_points)))
    return res, vol*np.ones_like(res)


def heston_path(theta_incoming, S0, T, N_points):
    r_annual, v0_annual, kappa, theta_annual, sigma_annual, rho_annual = theta_incoming
    r = r_annual/int(days_in_year_input.value)
    v0 = v0_annual/int(days_in_year_input.value)
    theta = theta_annual/int(days_in_year_input.value)
    sigma = theta_annual/np.sqrt(int(days_in_year_input.value))
    rho = rho_annual/np.sqrt(int(days_in_year_input.value))

    res = S0*np.ones(N_points+1)
    vh = np.ones(N_points+1)
    vh[0] = v0

    z1 = np.random.standard_normal(size=(N_points))

    for t in range(1, N_points+1):
        vh[t] = max(vh[t-1] + kappa*(theta-vh[t-1]) * (T/N_points) + sigma*np.sqrt(vh[t-1]) * np.sqrt(T/N_points)*z1[t-1], 0)

    z2 = rho*z1 + np.sqrt(1-rho**2)*np.random.standard_normal(size=(N_points))

    for t in range(1, N_points+1):
        res[t] = res[t-1]*np.exp((r - sigma**2/2) * (T/N_points) + np.sqrt(vh[t]) * np.sqrt(T/N_points) * z2[t-1])

    return res, np.sqrt(vh)


def GARCH_path(theta, S0, T, N_points):
    (mu, omega, alpha, beta) = theta
    points_per_day = N_points/T
    pmu = mu/points_per_day
    pomega = omega/np.sqrt(points_per_day)

    nu = np.random.standard_normal(2*N_points)
    ret = np.zeros(2*N_points)
    sigma = np.zeros(2*N_points)
    for i in range(1, 2*N_points):
        sigma[i] = np.sqrt(pomega + alpha*ret[i-1]**2 + beta*sigma[i-1]**2)
        ret[i] = pmu + sigma[i]*nu[i]

    retret = np.zeros(N_points+1)
    retret[1:] = ret[N_points:]
    return S0*np.cumprod(np.exp(retret/100)), sigma[N_points-1:]/100


def GJR_GARCH_path(theta, S0, T, N_points):
    (mu, omega, alpha, beta, gamma) = theta
    points_per_day = N_points/T
    pmu = mu/points_per_day
    pomega = omega/np.sqrt(points_per_day)

    nu = np.random.standard_normal(2*N_points)
    ret = np.zeros(2*N_points)
    sigma = np.zeros(2*N_points)
    for i in range(1, 2*N_points):
        sigma[i] = np.sqrt(pomega + alpha*ret[i-1]**2 + beta*sigma[i-1]**2 + gamma*int(ret[i-1] < 0)*ret[i-1]**2)
        ret[i] = pmu + sigma[i]*nu[i]

    retret = np.zeros(N_points+1)
    retret[1:] = ret[N_points:]
    return S0*np.cumprod(np.exp(retret/100)), sigma[N_points-1:]/100

models = {'Black-Scholes': {'params': [('annual risk-free IR r', -1, 1, 0.001, 0.024), ('annual volatility sigma', 0, 1, 0.001, 0.28)], 'path': BS_path},
          'GARCH(1,1)': {'params': [('excess log-return bias mu', -1, 1, 0.001, 0.049), ('volatility bias omega', -1, 1, 0.001, 0.026), ('alpha', -1, 1, 0.001, 0.102), ('beta', -1, 1, 0.001, 0.886)], 'path': GARCH_path},
          'GJR-GARCH(1,1)': {'params': [('excess log-return bias mu', -1, 1, 0.001, 0.002), ('volatility bias omega', -1, 1, 0.001, 0.027), ('alpha', -1, 1, 0.001, 0), ('beta', -1, 1, 0.001, 0.9), ('asymmetry factor gamma', -1, 1, 0.001, 0.169)], 'path': GJR_GARCH_path},
          'Heston': {'params': [('annual risk-free IR r', -1, 1, 0.001, 0.024), ('v zero', 0, 1, 0.0001, 0.0276), ('kappa', 0, 5, 0.001, 1.200), ('theta', 0, 1, 0.001, 0.0660), ('sigma', 0, 1, 0.0001, 0.5928), ('rho', 0, 1, 0.0001, -0.6589)], 'path': heston_path},
          }
model_names = list(models.keys())
blocked_recompute = False
recompute_request = False

# Set up data
INITIAL_SPOT = 100

S = np.linspace(0.8*INITIAL_SPOT, 1.2*INITIAL_SPOT, 1001)
y = np.zeros_like(S)
source = ColumnDataSource(data=dict(x=S, curr=y, pnl_matu=y, pnl_int=y))

# Set up plot
plot = figure(plot_height=600, plot_width=750, title="Portfolio valuation", align="start", x_axis_label='Spot price', y_axis_label='Portfolio total value',
              tools="crosshair,pan,save,wheel_zoom",
              x_range=[0.8*INITIAL_SPOT, 1.2*INITIAL_SPOT], y_range=[-INITIAL_SPOT, INITIAL_SPOT])

plot.line('x', 'curr', source=source, line_width=4, line_alpha=0.8, muted_alpha=0.2, color=Spectral4[0], legend_label="Instantaneous value")
plot.line('x', 'pnl_int', source=source, line_width=3, line_alpha=1, muted_alpha=0.2, color=Spectral4[1], legend_label="Delayed value")
plot.line('x', 'pnl_matu', source=source, line_width=3, line_alpha=0.6, muted_alpha=0.2, color=Spectral4[3], legend_label="Payoff at maturity")

plot.title.align = 'center'
plot.title.text_font_size = '20pt'
plot.xaxis.axis_label_text_font_size = "20pt"
plot.yaxis.axis_label_text_font_size = "20pt"
plot.legend.location = "top_left"
plot.legend.click_policy = "mute"


curr_spot_input = Slider(title="Current spot value", value=INITIAL_SPOT, start=0, end=4*INITIAL_SPOT, step=0.1)
int_delay_input = Spinner(title="Intermediary value delay", value=1, low=0, high=5*365, step=1, width=140)

leg_toggle_names = []
qty_input = []
type_select = []
strike_input = []
maturity_input = []
asset_type_list = list(assets.keys())
for leg_nb in range(4):
    leg_toggle_names.append(f"Leg #{leg_nb}")
    qty_input.append(Spinner(title="Total quantity", low=-10**6, high=10**6, step=1, value=0, width=140, height=60))
    type_select.append(Select(title="Asset type:", options=asset_type_list, value=asset_type_list[leg_nb % len(asset_type_list)], width=140, height=60))
    strike_input.append(Spinner(title="Strike", low=0, high=1000, step=1, value=INITIAL_SPOT, width=140, height=60))
    maturity_input.append(Spinner(title="Time to maturity (days)", low=0, high=5*365, step=1, value=50, width=140, height=60))

leg_toggle = CheckboxButtonGroup(labels=leg_toggle_names, active=[0], sizing_mode="scale_width")

ask_recompute = Button(label="Launch computation!", button_type="success", width=200)
ask_recompute.disabled = True
instant_recompute = CheckboxGroup(labels=["Automatically recompute"], active=[0], width=180)
recompute_delay = Slider(title="Recompute delay (ms)", value=500, start=0, end=1000, step=10, width=200)

display_data = dict(param=['Fair price at inception', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho'],
                    th_value=[0 for i in range(6)],
                    emp_value=[0 for i in range(6)])
display_source = ColumnDataSource(data=display_data)

display_columns = [
        TableColumn(field="param", title=""),
        TableColumn(field="th_value", title="Theoritical value"),
        TableColumn(field="emp_value", title="Computed value"),
    ]
display_data_table = DataTable(source=display_source, columns=display_columns, width=600, height=200)


def update_UI(attrname, old, new):
    redraw_UI()


def update_UI_mono(new):
    redraw_UI()


def redraw_UI():
    delay_red_flag = False
    T_delay = int_delay_input.value
    for leg_nb in range(4):
        if(leg_nb not in leg_toggle.active):
            qty_input[leg_nb].disabled = True
            type_select[leg_nb].disabled = True
            strike_input[leg_nb].disabled = True
            maturity_input[leg_nb].disabled = True
        else:
            qty_input[leg_nb].disabled = False
            type_select[leg_nb].disabled = False
            strike_input[leg_nb].disabled = False
            maturity_input[leg_nb].disabled = False
        asset_type = type_select[leg_nb].value
        if(asset_type in ["Spot"]):
            strike_input[leg_nb].visible = False
            maturity_input[leg_nb].visible = False
        else:
            strike_input[leg_nb].visible = True
            maturity_input[leg_nb].visible = True
            if(leg_nb in leg_toggle.active):
                T = maturity_input[leg_nb].value
                delay_red_flag += (T < T_delay)

    if(delay_red_flag):
        int_delay_input.background = 'red'
    else:
        int_delay_input.background = 'white'

    if(instant_recompute.active):
        ask_recompute.disabled = True
        recompute_delay.disabled = False
    else:
        ask_recompute.disabled = False
        recompute_delay.disabled = True


def recompute_asked(new):
    recompute()


def param_changed(attrname, old, new):
    global blocked_recompute
    global recompute_request
    if(instant_recompute.active and not blocked_recompute and not recompute_request):
        recompute_request = True
        curdoc().add_timeout_callback(recompute, recompute_delay.value)


def param_changed_mono(new):
    global blocked_recompute
    global recompute_request
    if(instant_recompute.active and not blocked_recompute and not recompute_request):
        recompute_request = True
        curdoc().add_timeout_callback(recompute, recompute_delay.value)


def Monte_Carlo(S0, T, N_Tries):
    res = np.zeros((N_Tries, T+1))
    for i in range(N_Tries):
        res[i, :] = models[model_names[model_selection_input.active]]['path']([x.value for x in models[model_names[model_selection_input.active]]['params_inputs']], S0, T, T)[0]
    return res


def Adaptive_Monte_Carlo(S0, T, err, N_max_tries):
    res = []
    tries = 0
    flag = True
    while((flag and tries < N_max_tries) or tries < 10):
        tries += 1
        simu_path = models[model_names[model_selection_input.active]]['path']([x.value for x in models[model_names[model_selection_input.active]]['params_inputs']], S0, T, T)[0]
        res.append(simu_path)
        flag = ((np.std(res[:, -1])/S0) > err)
    return np.array(res)


def recompute():
    global blocked_recompute
    blocked_recompute = False

    global recompute_request
    recompute_request = False

    if(max([abs(qty_input[leg_nb].value) for leg_nb in range(4)]) <= 0):
        return

    # Generate the new curve
    S0 = curr_spot_input.value
    T_delay = int_delay_input.value
    S = np.concatenate([np.linspace(0, 0.8*S0, 20), np.linspace(0.8*S0, 1.2*S0, 51), np.linspace(1.2*S0, 2*S0, 20)])
    if(models[model_names[model_selection_input.active]]['resolution_method'].value == 'Analytical (recommended)'):
        S = np.linspace(0, 2*S0, 1001)
    close_index = np.argmin(abs(S-S0))
    maturity_value = np.zeros_like(S)
    current_value = np.zeros_like(S)
    int_value = np.zeros_like(S)

    th_greeks = np.zeros(6)
    emp_greeks = np.zeros(6)

    if(models[model_names[model_selection_input.active]]['resolution_method'].value == 'Analytical (recommended)'):
        r, vol = [x.value for x in models[model_names[0]]['params_inputs']]

        for leg_nb in range(4):
            Q = qty_input[leg_nb].value
            if(leg_nb in leg_toggle.active and Q != 0):
                T = maturity_input[leg_nb].value
                K = strike_input[leg_nb].value
                asset_type = type_select[leg_nb].value

                T_int = max(T-T_delay, 0)
                unit_val = assets[asset_type]["BS_analytical"](S, K, vol, T, r)[0]
                unit_val_int = assets[asset_type]["BS_analytical"](S, K, vol, T_int, r)[0]
                th_greeks += Q*np.array(assets[asset_type]["BS_analytical"](S0, K, vol, T, r))

                if(T <= 0):
                    unit_val = assets[asset_type]["payoff"](S, K)
                if(T_int <= 0):
                    unit_val_int = assets[asset_type]["payoff"](S, K)

                current_value += Q*unit_val
                int_value += Q*unit_val_int
                maturity_value += Q*assets[asset_type]["payoff"](S, K)
    else:
        T_max = max([maturity_input[leg_nb].value for leg_nb in range(4)])
        monte_carlo_means = np.zeros_like(S)
        monte_carlo_std = np.zeros_like(S)
        monte_carlo_nb = np.zeros_like(S)

        for leg_nb in range(4):
            Q = qty_input[leg_nb].value
            if(leg_nb in leg_toggle.active and Q != 0):
                K = strike_input[leg_nb].value
                asset_type = type_select[leg_nb].value
                maturity_value += Q*assets[asset_type]["payoff"](S, K)

        for i, s_start in enumerate(S):
            if(not i % 10):
                print('Computing for s=',  s_start)
            sim = np.zeros((1, 1))
            if((models[model_names[model_selection_input.active]]['resolution_method'].value == 'Monte-Carlo')):
                sim = Monte_Carlo(s_start, T_max, 1000*(1+2*int(abs(close_index-i) < 2)))
            else:
                sim = Adaptive_Monte_Carlo(s_start, T_max, 0.02, 1000*(1+2*int(abs(close_index-i) < 2)))

            for leg_nb in range(4):
                Q = qty_input[leg_nb].value
                if(leg_nb in leg_toggle.active and Q != 0):
                    T = maturity_input[leg_nb].value
                    K = strike_input[leg_nb].value
                    asset_type = type_select[leg_nb].value

                    T_int = max(T-T_delay, 0)
                    res = assets[asset_type]["payoff"](sim[:, T], K)
                    res_int = assets[asset_type]["payoff"](sim[:, T_int], K)

                    unit_val = np.mean(res)
                    monte_carlo_means[i] = np.mean(res)
                    monte_carlo_std[i] = np.std(res)
                    monte_carlo_nb[i] = len(res)

                    unit_val_int = np.mean(res_int)

                    if(T <= 0):
                        unit_val = assets[asset_type]["payoff"](S, K)
                    if(T_int <= 0):
                        unit_val_int = assets[asset_type]["payoff"](S, K)

                    current_value[i] += Q*unit_val
                    int_value[i] += Q*unit_val_int

        # print(monte_carlo_means)
        # print(monte_carlo_std)
        # print(monte_carlo_nb)

    # GREEKS
    emp_greeks[0] = current_value[close_index]
    emp_greeks[1] = (current_value[close_index+1] - current_value[close_index-1])/(S[close_index+1]-S[close_index-1])
    emp_greeks[2] = (current_value[close_index+1] + current_value[close_index-1] - 2*current_value[close_index])/((S[close_index+1]-S[close_index])**2)
    emp_greeks[3] = (int_value[close_index]-current_value[close_index])/T_delay

    current_price = current_value[np.argmin(abs(S-S0))]
    curr_pnl = current_value-current_price
    matu_pnl = maturity_value-current_price
    int_pnl = int_value-current_price

    plot.x_range.start = 0.8*S0
    plot.x_range.end = 1.2*S0
    i_low, i_high = np.argmax(S > plot.x_range.start), np.argmin(S < plot.x_range.end)
    y_lowbound = min(np.min(current_value[i_low:i_high+1]), np.min(maturity_value[i_low:i_high+1]), np.min(int_value[i_low:i_high+1]))
    y_highbound = max(np.max(current_value[i_low:i_high+1]), np.max(maturity_value[i_low:i_high+1]), np.max(int_value[i_low:i_high+1]))
    if(y_highbound-y_lowbound > 0):
        plot.y_range.start = y_lowbound
        plot.y_range.end = y_highbound

    source.data = dict(x=S, curr=current_value, pnl_matu=maturity_value, pnl_int=int_value)
    display_source.data['th_value'] = np.round(th_greeks, 3)
    if(models[model_names[model_selection_input.active]]['resolution_method'].value != 'Analytical (recommended)'):
        display_source.data['th_value'] = ['N/A']*6
    display_source.data['emp_value'] = list(np.round(emp_greeks[:4], 3))+['Not computed']*2

to_link = [curr_spot_input, int_delay_input]
for leg_nb in range(4):
    to_link += [type_select[leg_nb], qty_input[leg_nb], strike_input[leg_nb]]
for w in to_link:
    w.on_change('value', param_changed)

leg_toggle.on_click(param_changed_mono)
leg_toggle.on_click(update_UI_mono)

instant_recompute.on_click(update_UI_mono)

ask_recompute.on_click(recompute_asked)

for w in [int_delay_input]+[type_select[leg_nb] for leg_nb in range(4)]:
    w.on_change('value', update_UI)

# Set up layouts and add to document
leg_inputs = [column(qty_input[i], type_select[i], strike_input[i], maturity_input[i], sizing_mode="scale_both") for i in range(4)]
inputs = column(row(curr_spot_input, int_delay_input),
                column(leg_toggle, row(leg_inputs[0], leg_inputs[1], leg_inputs[2], leg_inputs[3])),
                row(ask_recompute, instant_recompute, recompute_delay),
                row(display_data_table), width=750)

tab1 = Panel(child=row(inputs, plot), title='Overview')

source_underlying_example = ColumnDataSource(data=dict(T=[], S=[], sig=[]))

example_plot = figure(plot_height=400, plot_width=800, title="Underlying example path", x_axis_label='Time', y_axis_label='Underlying price',
                      tools="crosshair,save",
                      x_range=[0, 1], y_range=[90, 110], align='start')

example_plot.line('T', 'S', source=source_underlying_example, line_width=4, line_alpha=0.8, color=Spectral4[0])

example_vol_plot = figure(plot_height=250, plot_width=800, title="Underlying example volatility", y_axis_label='Underlying volatility',
                          tools="crosshair,save",
                          x_range=[0, 1], y_range=[0, 1], align='start')

example_vol_plot.line('T', 'sig', source=source_underlying_example, line_width=4, line_alpha=0.8, color=Spectral4[3])

example_plot.title.align = 'center'
example_plot.title.text_font_size = '20pt'
example_plot.xaxis.axis_label_text_font_size = "20pt"
example_plot.yaxis.axis_label_text_font_size = "20pt"

example_vol_plot.title.align = 'center'
example_vol_plot.title.text_font_size = '15pt'
example_vol_plot.yaxis.axis_label_text_font_size = "15pt"

blocked_clear = False


def clear_display():
    global blocked_clear
    if(not blocked_clear):
        display_zone.text = ''


def clear_text_asked(attrname, old, new):
    clear_display()


def fit_from_file_asked(attrname, old, new):
    if(len(new)):
        fit_model_from_file()
        fit_from_file_button.value = ''
        fit_from_file_button.filename = ''


def fit_model_from_file():
    try:
        print('Reading file and parsing data...')
        data = base64.b64decode(fit_from_file_button.value)
        df = pd.read_csv(BytesIO(data), sep=',', parse_dates=True, infer_datetime_format=True)
        ic = min([i for i in range(len(df.columns)) if is_numeric_dtype(df.dtypes[i])])
        print(f"First numerical column found is column #{ic} : {df.columns[ic]}")
        index = df.iloc[:, ic].to_numpy()
        returns = index[1:]/index[:-1]
        dilated_log_returns = 100*np.log(returns)
        prms = []

        if(model_names[model_selection_input.active] == 'GARCH(1,1)'):
            model = arch_model(dilated_log_returns).fit(disp='off')
            display_zone.text = str(model.summary())
            prms = model.params.to_numpy()

        elif(model_names[model_selection_input.active] == 'GJR-GARCH(1,1)'):
            model = arch_model(dilated_log_returns, o=1).fit(disp='off')
            display_zone.text = str(model.summary())
            prms = model.params.to_numpy()
            prms[3], prms[4] = prms[4], prms[3]

        elif(model_names[model_selection_input.active] == 'Black-Scholes'):
            prms = [int(days_in_year_input.value)*np.mean(returns-1), np.sqrt(int(days_in_year_input.value))*np.std(returns)]
            display_zone.text = ''

        global blocked_recompute
        blocked_recompute = True

        for w, x in zip(models[model_names[model_selection_input.active]]['params_inputs'], prms):
            w.value = np.round(x, 6)

        blocked_recompute = False
        model_parameters_changed_mono('')
        curdoc().add_timeout_callback(clear_display, 15000)
        print("Done!")

    except Exception as e:
        print("Model fiting critically failed:"+str(e))


def model_parameters_changed_mono(new):
    redraw_example()
    param_changed_mono(new)


def model_parameters_changed(attrname, old, new):
    redraw_example()
    param_changed(attrname, old, new)


def redraw_example():
    # Generate the new curve
    N_points = 1000
    S0 = curr_spot_input.value

    T_max = max([maturity_input[leg_nb].value for leg_nb in range(4) if(leg_nb in leg_toggle.active)])
    Time = np.linspace(0, T_max, N_points+1)

    val, vol = models[model_names[model_selection_input.active]]['path']([x.value for x in models[model_names[model_selection_input.active]]['params_inputs']], S0, T_max, N_points)

    example_plot.x_range.start = 0
    example_plot.x_range.end = T_max
    example_plot.y_range.start = 0.99*np.min(val)
    example_plot.y_range.end = 1.01*np.max(val)

    example_vol_plot.x_range.start = 0
    example_vol_plot.x_range.end = T_max
    example_vol_plot.y_range.start = 0.99*np.min(vol)
    example_vol_plot.y_range.end = 1.01*np.max(vol)

    source_underlying_example.data = dict(T=Time, S=val, sig=vol)

days_in_year_input = Select(title="Days in year:", options=['252', '365'], value='365', width=200)
days_in_year_input.on_change('value', model_parameters_changed)

for md in models:
    models[md]['params_inputs'] = []
    for nm, low_bound, high_bound, stepp, ini in models[md]['params']:
        models[md]['params_inputs'].append(Spinner(title=nm, low=low_bound, high=high_bound, step=stepp, value=ini, width=150))
        models[md]['params_inputs'][-1].on_change('value', model_parameters_changed)
    res_met = int(md == 'Black-Scholes')*['Analytical (recommended)']+['Monte-Carlo', 'Adaptive Monte-Carlo (experimental)']
    models[md]['resolution_method'] = Select(title="Resolution method:", options=res_met, value=res_met[0], width=150)
    models[md]['resolution_method'].on_change('value', param_changed)


model_selection_input = RadioButtonGroup(labels=model_names, active=0, width=600)
model_selection_input.on_click(model_parameters_changed_mono)

fit_desc = Div(text="""<h2><b>Fit model from file: </b><h2>""", align='center')
fit_from_file_button = FileInput(accept='.csv,.txt', align='center')
fit_from_file_button.on_change('value', fit_from_file_asked)

display_zone = PreText(text="""Default parameters are from CAC 40 returns""")

for w in [maturity_input[leg_nb] for leg_nb in range(4)]:
    w.on_change('value', model_parameters_changed)

tab2 = Panel(child=row(column(days_in_year_input,
                              model_selection_input,
                              row(children=[column(children=[models[md]['resolution_method']]+models[md]['params_inputs']) for md in models]),
                              row(fit_desc, fit_from_file_button, width=650),
                              row(display_zone, width=650),
                              width=650),
                       column(example_plot, example_vol_plot)
                       ), title='Underlying model')

curdoc().add_root(Tabs(tabs=[tab1, tab2]))
curdoc().title = "Moulagator"
redraw_UI()
redraw_example()
