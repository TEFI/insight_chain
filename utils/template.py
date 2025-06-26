from jinja2 import Environment

class TemplateWithSource:
    def __init__(self, name: str, env: Environment):
        self.name = name
        self.template = env.get_template(name)
        self.source, _, _ = env.loader.get_source(env, name)

    def render(self, **kwargs) -> str:
        return self.template.render(**kwargs)

    def __repr__(self):
        return f"<TemplateWithSource: {self.name}>"
