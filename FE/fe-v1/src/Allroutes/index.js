import { useRoutes } from "react-router-dom";
import { routes } from "../routes";
import toast, { Toaster } from 'react-hot-toast';
function Allroutes(){
    const elements = useRoutes(routes);
    return(
        <>
            <Toaster/>
            {elements}
        </>
    )
}
export default Allroutes;