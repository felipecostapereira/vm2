import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, date
import matplotlib.pyplot as plt
import os
import streamlit as st
import altair as alt
from scipy.optimize import curve_fit


def func_hiper(x, q0, n, a):
    return q0 / np.power(1+n*a*x,1/n)

companies = [
'YSUR=YSUR ENERGÍA ARGENTINA S.R.L.',
'YPF=YPF S.A.',
'WIN=WINTERSHALL ENERGIA S.A.',
'WDA=WINTERSHALL DEA ARGENTINA S.A',
'VST=VISTA ENERGY ARGENTINA SAU',
'VOG=Vista Oil & Gas Argentina SA',
'VNO=VENOIL S.A.',
'VIS=VISTA OIL & GAS ARGENTINA SAU',
'TPT=TECPETROL S.A.',
'TAU=TOTAL AUSTRAL S.A.',
'SHE=SHELL ARGENTINA S.A.',
'ROC=ROCH S.A.',
'PTRE=PETROLERA EL TREBOL S.A.',
'PLU=PLUSPETROL S.A.',
'PES=PATAGONIA ENERGY S.A.',
'PEL=PETROLERA ENTRE LOMAS S.A.',
'PCR=PETROQUIMICA COMODORO RIVADAVIA S.A.',
'PBE=PETROBRAS ARGENTINA S.A.',
'PAM=PAMPA ENERGIA S.A.',
'PAL=PAN AMERICAN ENERGY SL',
'PAE=PAN AMERICAN ENERGY (SUCURSAL ARGENTINA) LLC',
'OGDV=O&G DEVELOPMENTS LTD S.A.',
'MSA=MEDANITO S.A.',
'MAD=MADALENA AUSTRAL S.A.',
'KILW=KILWER S.A.',
'GREC=GRECOIL y CIA. S.R.L.',
'GPNE=GAS Y PETROLEO DEL NEUQUEN S.A.',
'ENE1=ENERGICON S.A.',
'EMEA=EXXONMOBIL EXPLORATION ARGENTINA S.R.L.',
'CNA=CAPETROL ARGENTINA S.A.',
'CHE=CHEVRON ARGENTINA S.R.L.',
'APS=CAPEX S.A.',
'APGA=PCO OIL AND GAS INTERNATIONAL INC (SUCURSAL A...',
'APEA=APACHE ENERGIA ARGENTINA S.R.L.',
'AME=AMERICAS PETROGAS ARGENTINA S.A.',
'AESA=ARGENTA ENERGIA S.A.',
'ACO=Petrolera Aconcagua Energia S.A.',
]
# companies = '\n\n'.join(companies)
# print(companies)

msg_help = {
    'vazao': 'Selcione Qo ou Qg para a vazão dos poços. Qo é em m³/dia e Qg em km³/dia.',
    'bloco': 'Selcione um ou mais blocos',
    'year': 'Considerar poços somente após esse ano',
    'poco': 'Selecione para mostrar no plot a nuvem de pontos',
    'envoltoria': 'Selcione para mostrar apenas a média e o intevalo de confiança P90,P10',
    'ajuste': 'Selcione tipo de ajuste ou nenhum (só harmonico por enquanto)',
    'L': 'Normalizar para L novo?',
    'q0': 'Vazão no início do poço (zero para deixar o modelo fitar)',
}

st.subheader('Vaca Muerta - Produção de Poço tipo')

path = os.path.join(os.getcwd(),'data')
df1 = pd.read_csv(os.path.join(path,'produccin-de-pozos-de-gas-y-petrleo-no-convencional_1.csv'), decimal='.')
df2 = pd.read_csv(os.path.join(path,'produccin-de-pozos-de-gas-y-petrleo-no-convencional_2.csv'), decimal='.')
dffrac = pd.read_csv(os.path.join(path,'datos-de-fractura-de-pozos-de-hidrocarburos-adjunto-iv-actualizacin-diaria.csv'), decimal='.')

dfprod = pd.concat([df1,df2], axis=0)
dfprod['data'] = dfprod['fecha_data'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
dfprod['qo(m3/d)'] = dfprod['prod_pet']/dfprod['fecha_data'].apply(lambda x:int(x[-2:]))
dfprod['qg(km3/d)'] = dfprod['prod_gas']/dfprod['fecha_data'].apply(lambda x:int(x[-2:]))

# colocando todos os poços na mesma data (m)
datazero = dfprod.groupby('idpozo')['data'].min()
dfprod = dfprod.merge(right=datazero, how='left', on='idpozo', suffixes=('', '_start'))
dfprod['m'] = dfprod['data'].dt.to_period('M').astype('int64') - dfprod['data_start'].dt.to_period('M').astype('int64')

# filtros
dfprod = dfprod[dfprod['formacion']=='vaca muerta']
dfprod = dfprod[dfprod['tipoestado']!='Parado Transitoriamente']

# filtros sidebar
st.sidebar.header('Filtros')
dfprod2 = dfprod
qoqg = st.sidebar.selectbox('Vazão: ', options=['qo(m3/d)', 'qg(km3/d)-nao funciona'], help=msg_help['vazao'])
# filtrando por bloco e ano de operação
sAreas = st.sidebar.multiselect('Bloco', dfprod2['areapermisoconcesion'].unique(), help=msg_help['bloco'])
year_start = st.sidebar.slider("Poços Depois de:", 2010, date.today().year, 2014, 1, help=msg_help['year'])
filtered_wells = dffrac[((dffrac['areapermisoconcesion'].isin(sAreas) & (dffrac['anio_if']>=year_start)))]['idpozo'].unique()
dfprod2 = dfprod2[(dfprod2['idpozo'].isin(filtered_wells)) ]
# sEmpresas = st.sidebar.multiselect('Emrpesas', dfprod2['empresa'].unique(), )

lenghts = dffrac[dffrac['idpozo'].isin(filtered_wells)]['longitud_rama_horizontal_m']
lenghts = lenghts[lenghts>0]
lmed = lenghts.mean()
st.sidebar.text(f'Lmédio: {lmed:.0f}m ({filtered_wells.shape[0]} poços)')

if sAreas:
    meses = np.arange(0,360,1)
    max_x = 240
    wells = st.sidebar.checkbox('Poços', value=False, help=msg_help['poco'])
    envoltoria = st.sidebar.checkbox('Envoltoria', value=True, help=msg_help['envoltoria'])
    fit = st.sidebar.selectbox('Ajuste', options=['Nenhum','Hiperbolico','Exponencial'], help=msg_help['ajuste'])

    if fit == 'Hiperbolico':
        st.sidebar.latex(r'''Q(t) = \frac{Q_0}{(1+nat)^\frac{1}{n}}''')
    elif fit == 'Exponencial':
        st.sidebar.latex(r'''Q(t) = Q_0 \exp(a)''')

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    if wells:
        sns.lineplot(data=dfprod2, x='m', hue='idpozo', y=qoqg, palette='tab20', alpha=0.6, ax=ax, legend=False)

    if envoltoria:
        sns.lineplot(data=dfprod2, x='m', y=qoqg, label='média', estimator='mean', errorbar=('pi',80), ax=ax)

    if fit == 'Hiperbolico':
        pmed = dfprod2.groupby('m')[qoqg].mean()
        p10 = dfprod2.groupby('m')[qoqg].quantile(0.9)
        p90 = dfprod2.groupby('m')[qoqg].quantile(0.1)
        pmed = pmed[pmed>5]
        p10=p10[p10>5]
        p10=p10[p90>5]

        cols = st.columns([3,2,2,2])
        with cols[0]:
            L = st.number_input(f"L(m) do DP (Lmed = {lmed:.0f})", value=lmed, step=100.0, help=msg_help['L'])
        with cols[1]:
            q0_med = st.number_input(r"$Q_0$ Pmed", value=int(np.max(pmed)), step=10, help='Vazão no início do poço (zero para deixar livre)')
        with cols[2]:
            q0_p10 = st.number_input(r"$Q_0$ P10 ", value=int(np.max(p10)), step=10,)
        with cols[3]:
            q0_p90 = st.number_input(r"$Q_0$ P90 ", value=int(np.max(p90)), step=10,)
        # with cols[4]:
        logscale = st.checkbox('Escala log', value=False)

        x = pmed.index.values
        x_10 = p10.index.values
        x_90 = p90.index.values

        if q0_med >0: # quer usar um valor fixo de Q0?
            bounds_med = ([q0_med-0.0000001, -np.inf, -np.inf], [q0_med, np.inf, np.inf])
        else:
            bounds_med = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        if q0_p10 >0: bounds_p10 = ([q0_p10-0.0000001, -np.inf, -np.inf], [q0_p10, np.inf, np.inf])
        else: bounds_p10 = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
        if q0_p90 >0: bounds_p90 = ([q0_p90-0.0000001, -np.inf, -np.inf], [q0_p90, np.inf, np.inf])
        else: bounds_p90 = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

        params_med, _ = curve_fit(func_hiper, x, pmed, p0=[q0_med, 0.8, 0.8], bounds=bounds_med)
        params_p10, _ = curve_fit(func_hiper, x_10, p10, p0=[q0_p10, 0.8, 0.8], bounds=bounds_p10)
        params_p90, _ = curve_fit(func_hiper, x_90, p90, p0=[q0_p90, 0.8, 0.8], bounds=bounds_p90)

        prev_med = func_hiper(meses, *params_med)*L/lmed
        prev_p10 = func_hiper(meses, *params_p10)*L/lmed
        prev_p90 = func_hiper(meses, *params_p90)*L/lmed
        np_prev_med = prev_med.cumsum()*30.41
        np_prev_p10 = prev_p10.cumsum()*30.41
        np_prev_p90 = prev_p90.cumsum()*30.41

        sns.lineplot(data=prev_med[:max_x], ax=ax, lw=2, label=f'Ajuste {fit} - med', color='k')
        sns.lineplot(data=prev_p10[:max_x], ax=ax, lw=2, label=f'Ajuste {fit} - p10', color='g')
        sns.lineplot(data=prev_p90[:max_x], ax=ax, lw=2, label=f'Ajuste {fit} - p90', color='r')
        if logscale:
            ax.set_yscale('log')
        ax2 = ax.twinx()
        ax2.set_ylabel('Np(bbl)')
        sns.lineplot(data=np_prev_med[:max_x]*6.26, ax=ax2, lw=2, color='k', linestyle='--')
        sns.lineplot(data=np_prev_p10[:max_x]*6.26, ax=ax2, lw=2, color='g', linestyle='--')
        sns.lineplot(data=np_prev_p90[:max_x]*6.26, ax=ax2, lw=2, color='r', linestyle='--')

        ax.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
        ax.set_xlabel('Meses')

        with st.expander(':dart: Resultados (30 anos)'):
            dict_results = {
                'Caso': ['Pmed', 'P10', 'P90'],
                'Q0[m³/d]': [params_med[0],params_p10[0],params_p90[0]],
                'a': [params_med[2],params_p10[2],params_p90[2]],
                'n': [params_med[1],params_p10[1],params_p90[1]],
                'Np[kbbl]': [np_prev_med[-1]*6.29/1000,np_prev_p10[-1]*6.29/1000,np_prev_p90[-1]*6.29/1000],
                'L[m]': [lmed,lmed,lmed],
                'L[m] DP': [L,L,L],
                'fator': [L/lmed,L/lmed,L/lmed]
            }
            df_results = pd.DataFrame(dict_results)
            st.dataframe(
                df_results,
                column_config={
                    "Np[kbbl]": st.column_config.NumberColumn(format='%.0f'),
                    'L[m]':  st.column_config.NumberColumn(format='%.0f'),
                    'L[m] DP':  st.column_config.NumberColumn(format='%.0f'),
                    'fator':  st.column_config.NumberColumn(format='%.2f'),
                },
                hide_index=True,
            )

    st.pyplot(ax.figure, clear_figure=True)
