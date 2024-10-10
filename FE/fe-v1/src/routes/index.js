import AuthUser from "../client/components/AuthUser";
import AuthLayout from "../client/layouts/AuthLayout";
import Chat from "../client/pages/Chat";
import Home from "../client/pages/Home";
import Login from "../client/pages/Login";
import Register from "../client/pages/Register";

export const routes = [
    {
        path:"/users/register",
        element:<AuthLayout><Register/></AuthLayout>
    },
    {
        path:"/users/login",
        element: <AuthLayout><Login/></AuthLayout>
    },
    {
        path:"/",
        element:<AuthUser/>,
        children:[
            {
                path:"/",
                element:<Home/>
            },
            {
                path:"chat/:id",
                element:<Chat/>
            }

        ]
    }
];
