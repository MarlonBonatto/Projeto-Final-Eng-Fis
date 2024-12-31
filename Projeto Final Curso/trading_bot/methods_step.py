import os
import logging
import pandas as pd
import numpy as np

from tqdm import tqdm
from .utils import (
    format_currency,
    format_position,
    get_data_with_date
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, ws=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, ws + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = 0
        next_state = get_state(data, t + 1, ws + 1)

        # select an action
        action = agent.act(state)


        # BUY                                                                                                                                
        if action == 1 and len(agent.inventory) < 1:                                                 
            agent.inventory.append(data[t])                                                           

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price 
            reward = delta
            total_profit += delta

        # HOLD
        else:
            pass

        # rendimento da posição aberta
        delta_dia = 0
        if len(agent.inventory) > 0:
            bought_price = agent.inventory[0]
            delta_dia = data[t] - bought_price 
            reward += delta_dia

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 3 == 0:    #quantidade de ep para salvar
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, data_with_date, ws, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    negociacoes = []
    datas_negociacoes = []
    agent.inventory = []
    operação = []
    list_date = []

    state = get_state(data, 0, ws + 1)

    profit_df = pd.DataFrame(columns=["Date", "Profit"])
 
    for t in range(data_length):        
        reward = 0
        next_state = get_state(data, t + 1, ws + 1)
        
        # select an action
        action = agent.act(state, is_eval=True)
        
            
        # BUY
        if action == 1 and len(agent.inventory) < 1:
            agent.inventory.append(data[t])
            operação.append('BUY')
            date = pd.to_datetime(data_with_date[t])
            list_date.append(date)
            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta
            total_profit += delta
            negociacoes.append((delta))
            date = pd.to_datetime(data_with_date[t])
            datas_negociacoes.append(date)
            list_date.append(date)
            operação.append('SELL')
            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))
            operação.append('HOLD')
            date = pd.to_datetime(data_with_date[t])
            list_date.append(date)

        # rendimento da posição aberta
        delta_dia = 0
        if len(agent.inventory) > 0:
            bought_price = agent.inventory[0]
            delta_dia = data[t] - bought_price 
            reward += delta_dia    # reward step

        current_date = pd.to_datetime(data_with_date[t])  
        new_row = pd.DataFrame({"Date": [current_date], "Profit": [total_profit + delta_dia]})
        profit_df = pd.concat([profit_df, new_row], ignore_index=True)

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            reward = 0
            if len(agent.inventory) > 0:
                while len(agent.inventory) > 0:
                    bought_price = agent.inventory.pop(0)
                    delta = data[t] - bought_price
                    reward = delta 
                    total_profit += delta
                    negociacoes.append((delta))
                    date = pd.to_datetime(data_with_date[t])
                    datas_negociacoes.append(date)
                    if debug:
                        logging.debug("Sell at: {} | Position: {}".format(
                            format_currency(data[t]), format_position(data[t] - bought_price)))

            current_date = pd.to_datetime(data_with_date[t])  
            new_row = pd.DataFrame({"Date": [current_date], "Profit": [total_profit]})
            profit_df = pd.concat([profit_df, new_row], ignore_index=True)
            
            profit_df.to_csv("profit_by_date.csv", index=False)
            negociacoes_df = pd.DataFrame({'Date': datas_negociacoes,'Valores': negociacoes})
            negociacoes_df.to_csv("negociações_valores.csv", index=False)
            operações_df = pd.DataFrame({"Date": list_date, "Operação": operação})
            operações_df.to_csv("operação_dia.csv", index=False)
            return total_profit, history
