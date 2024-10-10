import { createStore, applyMiddleware, combineReducers } from 'redux';
import { thunk } from 'redux-thunk';
import UserReducer from './client/reducers/UserReducer';
import AccountReducer from './admin/reducers/AccountReducer';

const rootReducer = combineReducers({
    UserReducer,
    AccountReducer
});

const store = createStore(rootReducer, applyMiddleware(thunk));

export default store;
