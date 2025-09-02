import gitlab
import os
import time
import re

from soupsieve.util import lower


# Helper: create gitlab instance
def get_gitlab_instance(token, url):
    return gitlab.Gitlab(url, private_token=token)

# 1. Get user info
def get_user_info(gl):
    return gl.user

# 2. Search project or any text
def search_projects(gl, query):
    return gl.projects.list(search=query, iterator=True)

# 3. Get project info
def get_project_info(gl, project_id):
    return gl.projects.get(project_id)

# 4. Create project
def create_project(gl, name, namespace_id=None):
    data = {'name': name}
    if namespace_id:
        data['namespace_id'] = namespace_id
    return gl.projects.create(data)

# 5. Get files from project
def get_files_from_project(project, ref, path):
    return project.repository_tree(path=path, ref=ref, recursive=True, get_all=True)

# 6. Get list of pipeline
def get_pipelines(project):
    return project.pipelines.list()

# 7. Get list of pipeline for branch
def get_pipelines_for_branch(project, branch):
    return project.pipelines.list(ref=branch)

# 8. Get list of pipeline for Merge Request
def get_pipelines_for_mr(project, mr_iid):
    mr = project.mergerequests.get(mr_iid)
    return mr.pipelines()

# 9. Invoke pipeline for given branch
def run_pipeline_for_branch(project, branch):
    return project.pipelines.create({'ref': branch})

# 10. Invoke pipeline for given MR
def run_pipeline_for_mr(project, mr_iid):
    mr = project.mergerequests.get(mr_iid)
    return project.pipelines.create({'ref': mr.source_branch})

# 11. Create branch, modify .gitlab.yaml, commit, create MR, check pipeline, wait, merge MR
def automate_mr_pipeline(gl, project_id, new_branch, yaml_content, mr_title):
    project = gl.projects.get(project_id)
    master_branch = 'master'
    # Create branch
    project.branches.create({'branch': new_branch, 'ref': master_branch})
    # Get .gitlab-ci.yml file
    file_path = '.gitlab-ci.yml'
    file = project.files.get(file_path=file_path, ref=master_branch)
    # Update file in new branch
    file.content = yaml_content
    file.save(branch=new_branch, commit_message='Update .gitlab-ci.yml')
    # Create MR
    mr = project.mergerequests.create({
        'source_branch': new_branch,
        'target_branch': master_branch,
        'title': mr_title
    })
    # Run pipeline for MR
    pipeline = run_pipeline_for_mr(project, mr.iid)
    # Wait for pipeline to finish
    while True:
        pipeline.refresh()
        if pipeline.status in ['success', 'failed', 'canceled']:
            break
        time.sleep(10)
    # Merge MR if pipeline succeeded
    if pipeline.status == 'success':
        mr.merge()
    return pipeline.status

# 12. Check pipeline for master, if success create tag
def tag_on_success(gl, project_id, tag_name):
    project = gl.projects.get(project_id)
    pipelines = get_pipelines_for_branch(project, 'master')
    if pipelines and pipelines[0].status == 'success':
        project.tags.create({'tag_name': tag_name, 'ref': 'master'})
        return True
    return False

def file_name_filter(file_name, pattern):
    if not file_name:
        return False
    if not pattern:
        return True
    return re.search(pattern, file_name) is not None

def prepare_pattern(file_extension_expression):
    if not file_extension_expression:
        return None
    extensions = file_extension_expression.split('|')
    regular_expression = ''
    for ext in extensions:
        if regular_expression == '':
            regular_expression = re.escape(ext).replace(r'\*', '.*')
        else:
            regular_expression = regular_expression + '|' + str(re.escape(ext).replace(r'\*', '.*'))
    return r'(' + regular_expression + r')$'

# 13. Search text in all files of all projects in a group and subgroups, create report
def search_text_in_group(gl, group_id, search_text, ref, path, file_extension_expression):
    pattern = prepare_pattern(file_extension_expression)
    def process_group(group, parent_name=None):
        for subgroup in group.subgroups.list():
            process_group(gl.groups.get(subgroup.id), group.name)
        for project in group.projects.list():
            proj = gl.projects.get(project.id)
            print(f"Searching in project: {proj.name} ({proj.id})")
            files = get_files_from_project(proj, ref, path)
            for f in files:
                if f['type'] == 'blob' and file_name_filter(f['name'], pattern):
                    file_obj = proj.files.get(file_path=f['path'], ref='master')
                    content = file_obj.decode().splitlines()
                    for idx, line in enumerate(content, 1):
                        if lower(search_text) in lower(str(line)):
                            print({
                                'group': parent_name or group.name,
                                'project': proj.name,
                                'file': f['path'],
                                'line': idx,
                                'text': line
                            })
    process_group(gl.groups.get(group_id))

# Example usage (fill in your token and parameters)
if __name__ == "__main__":
    token = os.getenv('MY_GITLAB_TOKEN')
    url = os.getenv('MY_GITLAB_LINK')
    gl = get_gitlab_instance(token, url)
    # print(get_user_info(gl))
    # print(search_projects(gl, 'test'))
    # print(get_project_info(gl, <project_id>))
    # print(create_project(gl, 'new-project'))
    # project = gl.projects.get(<project_id>)
    # print(get_files_from_project(project))
    # print(get_pipelines(project))
    # print(get_pipelines_for_branch(project, 'master'))
    # print(get_pipelines_for_mr(project, <mr_iid>))
    # print(run_pipeline_for_branch(project, 'master'))
    # print(run_pipeline_for_mr(project, <mr_iid>))
    # print(automate_mr_pipeline(gl, <project_id>, 'feature-branch', 'content', 'MR Title'))
    # print(tag_on_success(gl, <project_id>, 'v1.0.0'))
    search_text_in_group(gl, 78223, 'Header', 'master', '/', '*.kt|*.java')
    pass
