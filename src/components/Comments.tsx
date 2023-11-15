import Giscus from "@giscus/react";
import { useEffect, useState } from "react";

const id = "inject-comments";

const Comments = () => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div id={id}>
      {mounted ? (
        <Giscus
          id={id}
          repo="yxian9/blog"
          repoId="R_kgDOKsrA-g"
          category="General"
          categoryId="DIC_kwDOKsrA-s4Ca53_"
          mapping="url"
          reactionsEnabled="1"
          emitMetadata="0"
          theme="dark_tritanopia"
          inputPosition="bottom"
          lang="en"
          // loading="lazy"
        />
      ) : null}
    </div>
  );
};

export default Comments;
